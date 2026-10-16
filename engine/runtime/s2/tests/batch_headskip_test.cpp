// G6_MTP_HEAD_SKIP (s2_batch.h head_skip_env): a solo DFlash2 round whose next round stays on DFlash2 runs the MTP
// first form without its head (Step::nohead) on all but every P-th such round. Host only, a fake engine whose drafts say
// which position they were drafted for (a right draft is the target token, a wrong one -1000 - position), so a draft
// used at another position than its own is caught. Checks, over the scenarios of batch_draftpar_test (serial and
// G6_DRAFT_PAR, K-stop off / 1 / 2, start on MTP, insert rule, G6_DFL_REUSE=2):
//   P = 1 (never skips) gives the same tokens, drafter requests, steps and stats as off (the drafter choice moved before
//   the first form changes nothing);
//   P = 4: the committed tokens are the target's; no verify or chain row carries a draft of another position (no stale
//   d[1]); no log-probability is read after a head-less step; head-less steps happen, only as first forms;
//   a second session admitted after a head-less round (demote) chains from the DFlash2 draft and both stay exact.
// Build: g++ -std=c++17 -O2 -I engine/runtime engine/runtime/s2/tests/batch_headskip_test.cpp -o /tmp/batch_headskip_test
#include "s2_batch.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
using namespace s2;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static int target(int pos) { return (pos * 37 + 11) % 997; }    // the token at position pos + 1
static int wrong(int pos) { return -1000 - pos; }               // a wrong draft for position pos

struct Fake
{
    int mtp = 0, dfl = 0;             // drafting quality: MTP 0 all right, 1 two in three, 2 none; DFlash2 depth = 7 - dfl
    std::vector<std::string> steps, reqs;
    std::map<int, std::vector<int>> got;   // committed tokens per session (position 12 + i)
    std::vector<int> last;            // the last draft or chain step's tokens (lp)
    bool nohead_last = false, pending = false;
    DflDrafts stash{};
    long nohead = 0, stale = 0, errors = 0, demote_after_skip = 0;
    int committed(int q) { return 12 + (int)got[q].size(); }
    Hooks hooks()
    {
        Hooks h;
        h.run = [this](const Step& s) {
            if (s.kind == Step::MAIN && pending) ++errors;
            if (s.nohead && s.kind != Step::DRAFT) ++errors;
            std::string l = s.kind == Step::MAIN ? "M" : s.kind == Step::CHAIN ? "C" : s.nohead ? "H" : "D";
            std::vector<int> out;
            for (size_t i = 0; i < s.sess.size(); ++i)
                for (int j = s.row0[i]; j < s.row0[i] + s.rows[i]; ++j)
                {
                    const Row& r = s.r[j];
                    // a draft row's token must be a draft for its own position: verify rows past the committed ones
                    // (r.pos), chain rows (r.pos + 1)
                    const int dp = s.kind == Step::MAIN ? r.pos : s.kind == Step::CHAIN ? r.pos + 1 : -1;
                    if (dp >= 0 && (s.kind == Step::CHAIN || r.pos >= committed(r.sess)) && r.tok != target(dp - 1) && r.tok != wrong(dp))
                        ++stale;
                    const bool ok = s.kind == Step::MAIN || mtp == 0 || (mtp == 1 && r.spos % 3 != 0);
                    out.push_back(s.nohead ? -7 : ok ? target(r.spos - 1) : wrong(r.spos));
                    l += " " + std::to_string(r.tok) + "@" + std::to_string(r.pos);
                }
            steps.push_back(l);
            if (s.kind != Step::MAIN) { last = out; nohead_last = s.nohead; }
            if (s.nohead) ++nohead;
            return out;
        };
        h.dfl_start = [](int) { return 0; };
        h.dfl = [this](const DflReq& r) {
            if (r.phase == 2)
            {
                if (!pending) ++errors;
                pending = false;
                return stash;
            }
            if (pending) ++errors;
            reqs.push_back(std::string(r.query ? "Q" : "C") + std::to_string(r.rows) + "@" + std::to_string(r.pos0) + " t" +
                           std::to_string(r.tok0) + " lo" + std::to_string(r.lo));
            DflDrafts o{};
            const int q0 = r.pos0 + r.rows;
            for (int i = 1; i < DFL_B; ++i) { o.tok[i] = i <= 7 - dfl ? target(q0 + i - 1) : wrong(q0 + i); o.lp[i] = -0.05f * i - 0.01f * (q0 % 7); }
            if (r.phase == 1)
            {
                pending = true;
                stash = o;
                DflDrafts g{};
                for (int i = 0; i < DFL_B; ++i) { g.tok[i] = -9; g.lp[i] = 1.f; }   // must not be used
                return g;
            }
            return o;
        };
        h.emit = [this](int q, const int* t, int k, int) { got[q].insert(got[q].end(), t, t + k); return false; };
        h.lp = [this](int row) {
            if (nohead_last) ++errors;     // a head-less step has no log-probabilities
            return row < (int)last.size() ? -0.02f * (float)((last[row] + 3000) % 11) : 0.f;
        };
        return h;
    }
};

struct Run
{
    std::vector<std::string> steps, reqs;
    std::map<int, std::vector<int>> got;
    Stats st;
    long nohead = 0, stale = 0, errors = 0, demote_after_skip = 0;
};

static bool exact(const std::map<int, std::vector<int>>& got)
{
    bool e = true;   // the sessions' own tokens are the target's (the insert's are its own)
    for (const auto& g : got)
        for (size_t i = 0; i < g.second.size(); ++i) e &= g.second[i] == target(11 + (int)i) || g.second[i] >= 900;
    return e;
}

// one scenario: dsel on, kstop mode km; phases of (mtp, dfl) quality, n rounds each; second: rounds of session 0 alone
// before a second session (no DFlash2) joins for 20 rounds (0 = none)
static Run scenario(int P, bool par, int km, bool start_mtp, bool insert, int reuse, int second = 0)
{
    if (par) setenv("G6_DRAFT_PAR", "1", 1);
    else unsetenv("G6_DRAFT_PAR");
    if (P) setenv("G6_MTP_HEAD_SKIP", std::to_string(P).c_str(), 1);
    else unsetenv("G6_MTP_HEAD_SKIP");
    Fake f;
    Batch b(2, 32, f.hooks());
    b.dflash(7, g6rt::SpecLen());
    DSel sel;
    sel.on = true;
    sel.dwell = 3;
    sel.start_mtp = start_mtp;
    sel.d.draft_round = 8.1; sel.m.draft_round = 5.5; sel.m.draft_each = 2.1;
    b.dsel(sel);
    if (km)
    {
        KStop k;
        k.mode = km; k.kmax = 3;
        b.kstop(k);
    }
    if (reuse) { DflReuse r; r.mode = reuse; r.min = 6; b.dfl_reuse(r); }
    Session s; s.prompt.assign(12, 5); s.first = 11; s.chunk = 8; s.n_gen = 100000; s.dflash = !reuse;
    if (insert) { s.ins.after = 40; s.ins.tokens = {900, 901, 902}; }
    b.admit({0}, {s});
    if (reuse) b.promote(0);
    if (second)
    {
        f.dfl = 0;
        for (int i = 0; i < second; ++i) CHECK(b.round({0}));
        const bool skipped = f.nohead_last;
        Session s2 = s; s2.dflash = false; s2.ins = g6rt::InsertRule();
        b.admit({1}, {s2});
        for (int i = 0; i < 20; ++i) CHECK(b.round({0, 1}));
        f.demote_after_skip = skipped;
    }
    else
    {
        const int phase[][3] = {{0, 5, 25}, {0, 0, 25}, {1, 4, 30}, {2, 1, 30}, {0, 6, 30}, {1, 2, 30}};
        for (const auto& p : phase)
        {
            f.mtp = p[0]; f.dfl = p[1];
            for (int i = 0; i < p[2]; ++i) CHECK(b.round({0}));
        }
    }
    Run r;
    r.steps = f.steps; r.reqs = f.reqs; r.got = f.got; r.st = b.stats(0);
    r.nohead = f.nohead; r.stale = f.stale; r.errors = f.errors + (f.pending ? 1 : 0); r.demote_after_skip = f.demote_after_skip;
    return r;
}

static bool same(const Run& a, const Run& c)
{
    return a.got == c.got && a.reqs == c.reqs && a.steps == c.steps && a.st.rounds == c.st.rounds && a.st.tokens == c.st.tokens &&
           a.st.mtp_rounds == c.st.mtp_rounds && a.st.dfl_rounds == c.st.dfl_rounds && a.st.switches == c.st.switches &&
           a.st.acc == c.st.acc && a.st.inserted == c.st.inserted;
}

int main()
{
    long cases = 0, skips = 0, rounds = 0, dfl_rounds = 0, demotes = 0;
    for (int par : {0, 1})
        for (int km : {0, 1, 2})
            for (int start_mtp : {0, 1})
                for (int insert : {0, 1})
                    for (int reuse : {0, 2})
                    {
                        const Run off = scenario(0, par, km, start_mtp, insert, reuse), one = scenario(1, par, km, start_mtp, insert, reuse),
                                  four = scenario(4, par, km, start_mtp, insert, reuse);
                        CHECK(same(off, one));
                        CHECK(off.nohead == 0 && one.nohead == 0 && one.st.head_skips == 0);
                        CHECK(off.stale == 0 && one.stale == 0 && four.stale == 0);
                        CHECK(off.errors == 0 && one.errors == 0 && four.errors == 0);
                        CHECK(exact(off.got) && exact(four.got));
                        CHECK(four.nohead == four.st.head_skips);
                        if (!same(off, one) || four.stale || four.errors)
                            std::printf("  par %d km %d start_mtp %d insert %d reuse %d: steps %zu/%zu, P=4 stale %ld errors %ld\n", par, km,
                                        start_mtp, insert, reuse, off.steps.size(), one.steps.size(), four.stale, four.errors);
                        ++cases; skips += four.st.head_skips; rounds += four.st.rounds; dfl_rounds += four.st.dfl_rounds;
                    }
    for (int par : {0, 1})
        for (int n = 20; n < 28; ++n)
        {
            const Run off = scenario(0, par, 1, false, false, 0, n), four = scenario(4, par, 1, false, false, 0, n);
            CHECK(off.stale == 0 && four.stale == 0 && off.errors == 0 && four.errors == 0);
            CHECK(exact(off.got) && exact(four.got) && four.got.count(1) && !four.got.at(1).empty());
            demotes += four.demote_after_skip;
            ++cases;
        }
    CHECK(skips > 0 && demotes > 0);
    std::printf("batch_headskip_test: %s (%ld cases; P=4: %ld of %ld rounds' first forms head-less, %ld DFlash2 rounds; %ld demotes "
                "right after a head-less round)\n", fails ? "FAIL" : "PASS", cases, skips, rounds, dfl_rounds, demotes);
    return fails ? 1 : 0;
}
