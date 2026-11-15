// Synthetic GGUF tensor ranges, real OS reads, no GPU or model. The test gate only pauses reads to force queue races.
#include "ds4_os.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using namespace strata::ds4;
using Clock = std::chrono::steady_clock;
std::atomic<size_t> direct_reads{0};
void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
struct ReadGate {
    std::mutex mu;
    std::condition_variable cv;
    bool armed = false, entered = false;
    void arm() {
        std::lock_guard<std::mutex> lk(mu);
        armed = true;
        entered = false;
    }
    void pause() {
        std::unique_lock<std::mutex> lk(mu);
        if (!armed) return;
        entered = true;
        cv.notify_all();
        cv.wait(lk, [&] { return !armed; });
    }
    void await() {
        std::unique_lock<std::mutex> lk(mu);
        check(cv.wait_for(lk, std::chrono::seconds(5), [&] { return entered; }), "reader did not reach gate");
    }
    void open() {
        std::lock_guard<std::mutex> lk(mu);
        armed = false;
        cv.notify_all();
    }
} gate;
struct OpenGateOnExit {
    ~OpenGateOnExit() { gate.open(); }
};
}

namespace strata::ds4::os {
inline bool gated_par(fd_t f, void* p, uint64_t n, uint64_t off) {
    gate.pause();
    return pread_par(f, p, n, off);
}
inline bool gated_direct(fd_t f, void* p, uint64_t n, uint64_t off) {
    gate.pause();
    const bool ok = pread_direct(f, p, n, off);
    if (ok) direct_reads.fetch_add(1, std::memory_order_relaxed);
    return ok;
}
}
// Include the real queue/store against pausable OS readers; no production test hooks or replacement store implementation.
#define pread_par gated_par
#define pread_direct gated_direct
#include "ds4_store.hpp"
#undef pread_par
#undef pread_direct

namespace {
void env(const char* key, const char* value) {
#if defined(_WIN32)
    check(_putenv_s(key, value ? value : "") == 0, "set environment");
#else
    check((value ? setenv(key, value, 1) : unsetenv(key)) == 0, "set environment");
#endif
}
struct Fixture {
    std::vector<ExpertStore::LayerGeo> geo;
    std::vector<os::fd_t> fds, dfds, pfds;
    std::filesystem::path dir;
    std::vector<size_t> slot, count;
    size_t budget = 0, capacity = 0, uniform = 0;
    explicit Fixture(const std::filesystem::path& path) : dir(path) {
        check(std::filesystem::create_directories(dir), "fixture directory must be new");
        std::vector<std::vector<uint8_t>> files(3);
        const int experts[] = {96, 48, 96, 64};
        const size_t matrix[] = {1213, 1213, 6007, 17011};
        for (int l = 0; l < 4; ++l) {
            ExpertStore::LayerGeo g;
            g.n_exp = experts[l];
            size_t at = 0;
            for (int r = 0; r < 3; ++r) {
                g.file[r] = r;
                g.off[r] = files[r].size() + 37 + 61 * l + 13 * r;
                g.bytes[r] = matrix[l] + 67 * r;
                g.stride[r] = g.bytes[r] + 19; // unaligned file starts and strides
                g.tr[r] = at;
                at += ((g.bytes[r] + 255) & ~size_t{255}) + ExpertStore::kSlack;
                files[r].resize((size_t) g.off[r] + (size_t) g.n_exp * (size_t) g.stride[r]);
            }
            g.blob = at;
            geo.push_back(g);
        }
        for (int r = 0; r < 3; ++r) {
            files[r].resize((files[r].size() + 8191) & ~size_t{4095});
            for (size_t i = 0; i < files[r].size(); ++i) files[r][i] = (uint8_t) ((i * 73 + (i >> 8) * 29 + r * 47) % 251);
            const auto p = dir / ("tensor-" + std::to_string(r) + ".gguf");
            std::ofstream out(p, std::ios::binary);
            out.write((const char*) files[r].data(), (std::streamsize) files[r].size());
            out.close();
            check((bool) out, "write synthetic GGUF");
            fds.push_back(os::open_ro(p.string()));
            dfds.push_back(os::open_direct(p.string()));
            pfds.push_back(os::open_par(p.string()));
            check(fds.back() != os::kBadFd && pfds.back() != os::kBadFd, "open synthetic GGUF");
        }
        std::map<size_t, size_t> groups;
        for (const auto& g : geo) groups[g.blob] += g.n_exp;
        size_t weight = 0;
        for (const auto& [blob, ne] : groups) {
            slot.push_back((blob + 2 * ExpertStore::kSlack + 4095) & ~size_t{4095});
            count.push_back(ne / 8);
            weight += ne * slot.back();
            capacity += count.back();
        }
        budget = weight / 8; // exact proportional split: 18, 12, 8 slots across three classes
        uniform = budget / slot.back();
    }
    ~Fixture() {
        gate.open();
        for (auto f : fds) os::close_fd(f);
        for (auto f : dfds) os::close_fd(f);
        for (auto f : pfds) os::close_fd(f);
        std::error_code ec;
        for (int r = 0; r < 3; ++r) std::filesystem::remove(dir / ("tensor-" + std::to_string(r) + ".gguf"), ec);
        std::filesystem::remove(dir, ec);
    }
    ExpertStore::Cfg cfg(ExpertStore::Policy policy = ExpertStore::kLru) const {
        ExpertStore::Cfg c;
        c.budget_bytes = budget;
        c.threads = 2;
        c.floor_bytes = 0;
        c.reserve = 0;
        c.policy = policy;
        return c;
    }
    void init(ExpertStore& s, const ExpertStore::Cfg& c, bool direct = true) const {
        check(s.init(c, geo, direct ? dfds : std::vector<os::fd_t>(3, os::kBadFd), pfds), "store init");
    }
    void bytes(const ExpertStore& s, int l, int e) const {
        const int sl = s.find(l, e);
        check(sl >= 0, "expert ready");
        for (int r = 0; r < 3; ++r) {
            std::vector<uint8_t> ref((size_t) geo[l].bytes[r]);
            check(os::pread_full(fds[r], ref.data(), ref.size(), geo[l].off[r] + e * geo[l].stride[r]), "reference read");
            check(std::memcmp(s.mat(sl, r), ref.data(), ref.size()) == 0, "matrix differs from file");
            check(s.mat(sl, r) == s.blob(sl) + geo[l].tr[r], "matrix layout changed");
        }
    }
    int get(ExpertStore& s, int l, int e, bool release = true) const {
        std::vector<char> ok;
        check(s.ensure(l, &e, 1, ok) && ok[0], "ensure expert");
        bytes(s, l, e);
        int sl = s.find(l, e);
        if (release) s.release(l, &e, 1, ok);
        return sl;
    }
};

void capacity_and_bytes(const Fixture& f, bool direct) {
    const size_t before_direct = direct_reads.load();
    ExpertStore s;
    f.init(s, f.cfg(), direct);
    check(s.class_count() == 3 && s.slots_max() == f.capacity, "per-class capacity");
    check(s.slots_max() > f.uniform, "classes must beat uniform capacity");
    size_t expected_bytes = 0;
    for (size_t i = 0; i < f.count.size(); ++i) expected_bytes += f.count[i] * f.slot[i];
    check(s.capacity_bytes() == expected_bytes && s.capacity_bytes() <= f.budget, "budget exceeded");
    check(s.min_slot_bytes() == f.slot.front() && s.slot_bytes() == f.slot.back(), "slot extrema");
    check(s.max_blob() == f.slot.back() - 2 * ExpertStore::kSlack, "maximum staging blob");
    // Initial commits leave holes in the GLOBAL slot id space (only 8 of the small class's 18 slots).
    check(s.slots_allocated() == 24, "one initial chunk per class");
    check(f.get(s, 3, 0) == 30, "global ids include uncommitted class tails");
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    size_t lo = 0;
    const int layer[] = {0, 2, 3};
    for (size_t ci = 0; ci < 3; ++ci) {
        for (int e = 0; e < (int) f.count[ci]; ++e) {
            int sl = f.get(s, layer[ci], e, false);
            check((size_t) sl >= lo && (size_t) sl < lo + f.count[ci], "wrong class slot id");
            uintptr_t a = (uintptr_t) s.blob(sl) - ExpertStore::kSlack;
            uintptr_t b = a + f.slot[ci];
            check(a >= s.arena_lo() && b <= s.arena_hi(), "slot outside arena");
            ranges.push_back({a, b});
        }
        lo += f.count[ci];
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) check(ranges[i - 1].second <= ranges[i].first, "overlapping slots");
    for (size_t ci = 0; ci < 3; ++ci)
        for (int e = 0; e < (int) f.count[ci]; ++e) f.bytes(s, layer[ci], e); // recheck after all neighboring reads
    check(s.slots_allocated() == s.slots_max(), "all class chunks grown");
    check(direct ? direct_reads.load() > before_direct : direct_reads.load() == before_direct, "direct/buffered read path not exercised");
    std::printf("PASS %s bytes/layout/ranges/capacity: %zu class slots > %zu uniform, %zu bytes <= %zu budget\n",
                direct ? "direct" : "buffered", s.slots_max(), f.uniform, s.capacity_bytes(), f.budget);
}

void eviction(const Fixture& f, ExpertStore::Policy policy) {
    ExpertStore s;
    auto c = f.cfg(policy);
    c.decay_every = 10000;
    f.init(s, c);
    f.get(s, 2, 0);
    f.get(s, 3, 0);
    for (int e = 0; e < 18; ++e) f.get(s, 0, e, e != 0); // oldest slot remains pinned
    s.touch(0, 1);
    f.get(s, 1, 0); // same size, different layer: shares small class
    check(s.find(0, 0) >= 0 && s.find(0, 1) >= 0, "pinned/touched expert evicted");
    check(s.find(0, 2) == -1, "policy did not choose coldest eligible expert");
    check(s.find(2, 0) >= 0 && s.find(3, 0) >= 0, "cross-class eviction");
    check(s.stats().evictions.load() == 1, "eviction statistics");
    for (int e = 1; e < 30; ++e) f.get(s, 1, e);
    check(s.find(0, 0) >= 0 && s.find(2, 0) >= 0, "pressure escaped class/pin");
    std::vector<char> ok(1, 1);
    int e = 0;
    s.release(0, &e, 1, ok);
    // Fill/pin the whole large class: it must fail rather than borrow another class's slots.
    for (e = 0; e < 8; ++e) f.get(s, 3, e, false);
    e = 8;
    check(!s.ensure(3, &e, 1, ok) && !ok[0] && s.stats().no_slot.load() > 0, "full pinned class borrowed slots");
    std::printf("PASS %s class-local eviction, touch, pins, no-slot statistics\n", policy == ExpertStore::kLru ? "LRU" : "LFU");
}

void guesses_and_put(const Fixture& f) {
    ExpertStore s;
    auto c = f.cfg();
    c.threads = 1;
    f.init(s, c, false);
    OpenGateOnExit cleanup;
    gate.arm();
    s.prefetch(0, 0);
    gate.await(); // first read blocked, all later guesses still queued
    s.prefetch(2, 0);
    s.prefetch(3, 0);
    s.cancel_stale(2); // cancels layers 0 and 2, keeps layer 3
    gate.open();
    s.preload({}, 0); // drains and finalizes every allocated class, including ids past nalloc
    check(s.find(0, 0) < 0 && s.find(2, 0) < 0, "stale guesses survived cancellation");
    f.get(s, 3, 0);
    check(s.stats().cancelled.load() == 2 && s.stats().prefetch_used.load() == 1, "prefetch/cancel statistics");
    for (int l = 0; l < 4; ++l) {
        const int e = 7;
        uint8_t* p = s.put_begin(l, e);
        check(p != nullptr && s.find(l, e) < 0 && s.put_begin(l, e) == nullptr, "put_begin visibility/duplicate");
        for (int r = 0; r < 3; ++r)
            check(os::pread_full(f.fds[r], p + f.geo[l].tr[r], f.geo[l].bytes[r], f.geo[l].off[r] + e * f.geo[l].stride[r]), "put reference bytes");
        s.put_done(l, e, true);
        f.bytes(s, l, e);
        check(s.put_begin(l, e + 1) != nullptr, "failed put slot");
        s.put_done(l, e + 1, false);
        check(s.find(l, e + 1) < 0, "failed put became ready");
        f.get(s, l, e + 1);
    }
    check(s.stats().put.load() == 8 && s.stats().fails.load() == 0, "put statistics");
    gate.arm();
    std::vector<int> es{30, 31}, slots;
    std::vector<char> ok;
    s.begin(2, es.data(), (int) es.size(), ok, slots);
    gate.await();
    check(!s.loaded(slots[0]) && s.find(2, es[0]) < 0, "begin published an unfinished read");
    gate.open();
    check(s.wait_slots(slots, ok, 0, 1) && ok[0] && !ok[1], "partial wait_slots");
    check(s.wait_slots(slots, ok, 1, 2) && ok[1] && s.loaded(slots[1]), "remaining wait_slots");
    f.bytes(s, 2, es[0]);
    f.bytes(s, 2, es[1]);
    s.release(2, es.data(), (int) es.size(), ok);
    std::printf("PASS deterministic prefetch/cancel_stale, begin/loaded/partial waits, put success/failure across classes\n");
}

void wait_any(const Fixture& f) {
    ExpertStore s;
    auto c = f.cfg();
    c.threads = 1;
    f.init(s, c, false);
    OpenGateOnExit cleanup;
    s.cancel_stale(3); // no guess ever started: nothing to cancel
    check(s.stats().cancelled.load() == 0, "cancel_stale without guesses");
    std::vector<int> es{40, 41}, slots;
    std::vector<char> ok;
    gate.arm();
    s.begin(1, es.data(), (int) es.size(), ok, slots);
    gate.await(); // the first read blocked, the second queued behind it (one reader)
    std::atomic<int> got{-2};
    std::thread w([&] { got.store(s.wait_any_loaded(slots, {1, 0})); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    check(got.load() == -2, "wait_any_loaded returned before any read landed");
    gate.open();
    w.join();
    check((got.load() == 0 || got.load() == 1) && s.loaded(slots[(size_t) got.load()]), "wait_any_loaded returned an unfinished slot");
    check(s.wait_slots(slots, ok, 0, 2) && ok[0] && ok[1], "wait_slots after wait_any_loaded");
    check(s.wait_any_loaded(slots, {1, 0}) == 1, "wait_any_loaded must keep the candidates' order");
    const std::vector<int> none{-1, slots[0]};
    check(s.wait_any_loaded(none, {0, 1}) == 0, "a candidate with no slot returns at once");
    check(s.wait_any_loaded(slots, {}) == -1, "no candidates");
    f.bytes(s, 1, es[0]);
    f.bytes(s, 1, es[1]);
    s.release(1, es.data(), (int) es.size(), ok);
    gate.arm();
    s.prefetch(3, 5); // a guess (its read held): cancel_stale scans again
    gate.await();
    s.cancel_stale(3);
    gate.open();
    s.preload({}, 0);
    check(s.find(3, 5) < 0 && s.stats().cancelled.load() == 1, "cancel_stale after a guess");
    std::printf("PASS wait_any_loaded (first landed, candidate order, no slot), cancel_stale with and without guesses\n");
}

void preload(const Fixture& f) {
    ExpertStore s;
    f.init(s, f.cfg());
    std::vector<std::pair<int, int>> list;
    for (int l : {0, 2, 3})
        for (int e = 0; e < 40; ++e) list.push_back({l, e}); // global prefix contains ONLY small experts
    check(s.preload(list, 0) == f.capacity, "preload did not use every class");
    const int layer[] = {0, 2, 3};
    for (int ci = 0; ci < 3; ++ci) {
        for (int e = 0; e < (int) f.count[ci]; ++e) f.bytes(s, layer[ci], e);
        check(s.find(layer[ci], (int) f.count[ci]) < 0, "preload selected colder expert");
    }
    check(s.preload(list, 0) == 0, "preload reread resident experts");
    // A populated cache still replaces cold experts, and protects resident members of the selected hot set.
    std::vector<std::pair<int, int>> hotter;
    for (int l : {0, 2, 3}) {
        hotter.push_back({l, 0});
        for (int e = 20; e < 60; ++e) hotter.push_back({l, e});
    }
    check(s.preload(hotter, 0) == f.capacity - 3, "preload must replace cold residents");
    for (int ci = 0; ci < 3; ++ci) {
        f.bytes(s, layer[ci], 0);
        for (int e = 20; e < 20 + (int) f.count[ci] - 1; ++e) f.bytes(s, layer[ci], e);
    }
    ExpertStore reserved;
    f.init(reserved, f.cfg());
    check(reserved.preload(list, 7) == f.capacity - 7, "per-class preload reserve");
    std::printf("PASS hottest-first preload across full classes, cold replacement, resident protection, exact reserve\n");
}

void failed_read(const Fixture& f) {
    ExpertStore s;
    auto bad = std::vector<os::fd_t>(3, os::kBadFd);
    check(s.init(f.cfg(), f.geo, bad, bad), "failed-read store init");
    int e = 0;
    std::vector<char> ok;
    check(!s.ensure(2, &e, 1, ok) && !ok[0] && s.find(2, e) < 0, "failed read published bytes");
    check(s.stats().fails.load() == 1, "failed read statistics");
    std::vector<std::pair<int, int>> failed_preload;
    for (int e2 = 1; e2 < 13; ++e2) failed_preload.push_back({2, e2});
    s.preload(failed_preload, 0);
    s.set_pfds(f.pfds);
    for (e = 0; e < 12; ++e) f.get(s, 2, e, false);
    check(!s.ensure(2, &e, 1, ok) && !ok[0], "failed preload damaged pin counts");
    check(s.stats().misses.load() == 14 && s.stats().bytes_read.load() > 0, "demand statistics");
    std::printf("PASS failed read cleanup, reopened fallback handles, class refill, failure/miss/read statistics\n");
}

void mapping_and_shrink(const Fixture& f, bool identity) {
    std::map<uintptr_t, std::pair<size_t, uintptr_t>> mapped;
    size_t maps = 0, unmaps = 0;
    {
        ExpertStore s;
        auto c = f.cfg();
        c.reserve = 7;
        c.map_chunk = [&](void* p, size_t n) -> void* {
            uintptr_t h = (uintptr_t) p, d = h + (identity ? 0 : 0x10000000);
            check(mapped.emplace(h, std::make_pair(n, d)).second, "chunk mapped twice");
            ++maps;
            return (void*) d;
        };
        c.unmap_chunk = [&](void* p) {
            check(mapped.erase((uintptr_t) p) == 1, "wrong chunk unmap");
            ++unmaps;
        };
        f.init(s, c);
        check(s.gpu_mapped() && s.identity_mapped() == identity, "mapping flags");
        for (int l : {0, 2, 3})
            for (int e = 0; e < (l == 0 ? 18 : l == 2 ? 12 : 8); ++e) {
                int sl = f.get(s, l, e);
                uintptr_t b = (uintptr_t) s.blob(sl);
                auto ch = mapped.upper_bound(b);
                check(ch != mapped.begin(), "mapped chunk lookup");
                --ch;
                check(b < ch->first + ch->second.first, "slot outside mapped chunk");
                check(s.dev_blob(sl) == ch->second.second + b - ch->first, "wrong class device offset");
            }
        std::vector<ExpertStore::Vis> ev;
        s.sync(ev, [](int, int) { return true; });
        check(s.visible_count() == f.capacity - 7, "GPU reserve must be split across classes");
        check(!ev.empty(), "missing visibility events");
        // Every class retains evictable capacity even while others remain GPU-visible.
        for (int l : {0, 2, 3}) f.get(s, l, 40);
        for (int l : {0, 2, 3})
            for (int e = 0; e < f.geo[l].n_exp; ++e) s.unmark(l, e);
        check(s.visible_count() == 0, "unmark");
        f.get(s, 0, 17, false); // last small-class slot is in its last chunk
        const size_t before = s.slots_allocated();
        check(s.shrink_chunks(1) == 0 && s.slots_allocated() == before, "shrink ignored pinned tail");
        int e = 17;
        s.release(0, &e, 1, std::vector<char>(1, 1));
        check(s.shrink_chunks(1) == 1 && s.slots_allocated() == before - 2, "wrong class chosen to shrink");
        check(s.shrink_chunks(100) == 5 && s.slots_allocated() == 0 && !s.gpu_mapped(), "shrink every class");
        for (int l : {0, 2, 3}) f.get(s, l, 60);
        check(s.slots_allocated() == 24 && s.gpu_mapped(), "refill class chunks after shrink");
        check(s.shrunk() == 6, "shrink statistics");
    }
    check(mapped.empty() && maps == unmaps, "mapping lifetime leak");
    std::printf("PASS %s mapping/dev_blob/sync/unmark, busy-tail shrink, all-class shrink/refill, balanced callbacks\n",
                identity ? "identity" : "translated");
}

void growth_and_legacy(const Fixture& f) {
    for (size_t budget : {size_t{1}, f.slot[0] - 1, f.slot.back(), f.budget / 2 + 123, f.budget + 557}) {
        ExpertStore fractional;
        auto fc = f.cfg();
        fc.budget_bytes = budget;
        fc.ram_avail = [] { return uint64_t{0}; };
        size_t expected = 0, bytes = 0;
        for (size_t ci = 0; ci < 3; ++ci) {
            const size_t share = budget * (f.count[ci] * 8 * f.slot[ci]) / (f.budget * 8);
            const size_t n = share / f.slot[ci];
            expected += n;
            bytes += n * f.slot[ci];
        }
        check(fractional.init(fc, f.geo, f.dfds, f.pfds) == (expected > 0), "fractional budget init");
        check(fractional.slots_max() == expected && fractional.capacity_bytes() == bytes && bytes <= budget, "fractional budget rounding");
    }
    auto c = f.cfg();
    uint64_t available = 0;
    c.floor_bytes = 1024;
    c.ram_avail = [&] { return available; };
    ExpertStore s;
    f.init(s, c);
    check(s.slots_allocated() == 0, "grew below floor");
    available = 8 * f.slot[0] + c.floor_bytes - 1;
    int e = 0;
    std::vector<char> ok;
    check(!s.ensure(0, &e, 1, ok) && s.slots_allocated() == 0, "floor boundary exceeded");
    ++available;
    f.get(s, 0, 0);
    check(s.slots_allocated() == 8, "exact floor boundary did not grow");
    available = ~uint64_t{0};
    f.get(s, 3, 0);
    check(s.slots_allocated() == 16, "unknown available RAM should permit growth");
    env("DS4_STORE_CLASSES", "0");
    {
        ExpertStore old;
        f.init(old, f.cfg());
        check(old.class_count() == 1 && old.slot_bytes() == f.slot.back() && old.slots_max() == f.uniform, "legacy slot size/count");
        check(old.arena_hi() - old.arena_lo() == ((f.uniform + 7) / 8) * 8 * f.slot.back(), "legacy reservation size");
        env("DS4_STORE_CLASSES", "1"); // init reads once; later environment changes cannot change the layout
        std::vector<int> ids;
        for (int i = 0; i < (int) f.uniform; ++i) {
            int sl = f.get(old, i % 4, i, false);
            check(sl == i, "legacy global id order");
            if (i > 0) check(old.blob(sl) - old.blob(ids.back()) == (ptrdiff_t) f.slot.back(), "legacy uniform spacing");
            ids.push_back(sl);
        }
        check(old.slots_allocated() == f.uniform, "legacy growth count");
        check(old.class_count() == 1, "classes setting reread after init");
    }
    env("DS4_STORE_CLASSES", nullptr);
    env("DS4_STORE_PREGROW", "1");
    {
        ExpertStore pregrown;
        f.init(pregrown, f.cfg());
        check(pregrown.slots_allocated() == f.capacity, "pregrow did not grow all classes");
    }
    env("DS4_STORE_PREGROW", "0");
    std::printf("PASS fractional budgets, RAM floor/growth/pregrow, legacy exact size/count/range/spacing, read-once flag\n");
}

void partial_mapping(const Fixture& f) {
    size_t attempts = 0, unmaps = 0;
    {
        ExpertStore s;
        auto c = f.cfg();
        c.map_chunk = [&](void* p, size_t) -> void* { return ++attempts == 1 ? nullptr : p; };
        c.unmap_chunk = [&](void*) { ++unmaps; };
        f.init(s, c);
        check(s.gpu_mapped() && s.identity_mapped(), "later class mapping hidden by first failure");
        int small = f.get(s, 0, 0), medium = f.get(s, 2, 0), large = f.get(s, 3, 0);
        check(s.dev_blob(small) == 0 && s.dev_blob(medium) != 0 && s.dev_blob(large) != 0, "partial class device addresses");
        std::vector<ExpertStore::Vis> ev;
        s.sync(ev, [](int, int) { return true; });
        check(s.visible_count() == 2 && ev.size() == 2, "unmapped class became visible");
        check(s.shrink_chunks(10) == 1 && s.slots_allocated() == 16, "shrink did not stop at visible chunk");
        s.unmark(2, 0);
        s.unmark(3, 0);
        check(s.shrink_chunks(10) == 2 && !s.gpu_mapped(), "unmarked mapped chunks did not shrink");
    }
    check(attempts == 3 && unmaps == 2, "failed mapping was unregistered");
    std::printf("PASS partial map failure, later-class GPU mapping, visible-tail shrink barrier\n");
}

void wakeup(const Fixture& f) {
    IoQueue q(1);
    OpenGateOnExit cleanup;
    std::atomic<int> left{1};
    std::atomic<bool> bad{false};
    std::vector<uint8_t> buf(32);
    IoQueue::Job j;
    j.fd = f.pfds[0];
    j.dst = buf.data();
    j.len = buf.size();
    j.left = &left;
    j.bad = &bad;
    gate.arm();
    q.push(&j, 1);
    gate.await();
    const uint64_t seen = q.done_seen(); // same ordering as wait_slots
    check(left.load(std::memory_order_acquire) == 1, "completion happened before condition check");
    gate.open();
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (q.done_seen() == seen && Clock::now() < deadline) std::this_thread::yield();
    check(q.done_seen() != seen && left.load(std::memory_order_acquire) == 0 && !bad.load(), "forced completion failed");
    const auto start = Clock::now();
    q.wait_change(seen, 500000); // COMPLETED between check and wait: must not wait the 500 ms timeout
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    check(ms < 100, "lost completion caused timeout");
    // Completion still pending when wait begins: notification must wake it too.
    left.store(1);
    gate.arm();
    q.push(&j, 1);
    gate.await();
    const uint64_t second = q.done_seen();
    std::thread opener([] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); gate.open(); });
    q.wait_change(second, 500000);
    opener.join();
    check(left.load(std::memory_order_acquire) == 0, "pending completion did not wake waiter");
    q.wait_any(1); // existing API remains available
    std::printf("PASS forced completion between check/wait: %.3f ms (500 ms timeout); pending completion wakes\n", ms);
}
}

int main(int argc, char** argv) {
    try {
        env("DS4_STORE_CLASSES", nullptr); // default-on path
        env("DS4_STORE_CHUNK_SLOTS", "8");
        env("DS4_STORE_PREGROW", "0");
        Fixture f(argc > 1 ? argv[1] : "store-classes-fixtures");
        capacity_and_bytes(f, true);
        capacity_and_bytes(f, false);
        eviction(f, ExpertStore::kLru);
        eviction(f, ExpertStore::kLfu);
        guesses_and_put(f);
        wait_any(f);
        preload(f);
        failed_read(f);
        mapping_and_shrink(f, true);
        mapping_and_shrink(f, false);
        growth_and_legacy(f);
        partial_mapping(f);
        wakeup(f);
        std::printf("ALL STORE SIZE-CLASS TESTS PASSED\n");
        return 0;
    } catch (const std::exception& e) {
        gate.open();
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
