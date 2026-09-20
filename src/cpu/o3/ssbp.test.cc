/*
 * Copyright (c) 2004-2005 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include <string>

#include "base/types.hh"
#include "cpu/o3/ssbp.hh"
#include "params/SSBP.hh"

using namespace gem5;
using gem5::o3::SSBP;

namespace
{

/** Two load PCs that differ inside the low 12 bits, so they select
 *  different table entries.
 */
constexpr Addr PcA = 0x1000;
constexpr Addr PcB = 0x1040;

/** A third PC that deliberately collides with PcA.  getIndex() is
 *  load_PC & (numEntries - 1), so any two PCs sharing their low 12 bits
 *  share an entry however far apart they are.
 */
constexpr Addr PcCollidesWithA = 0x2000;

/** Table size used by every test.  Must be a power of two or
 *  SSBP::init() calls fatal().
 */
constexpr int NumEntries = 4096;

/** Alias-check granularity, matching the LSQDepCheckShift default.  None
 *  of the tests below reach the overlap test, so the value only has to be
 *  a legal one.
 */
constexpr unsigned DepCheckShift = 4;

/**
 * Builds an SSBP without a CPU.
 *
 * SSBP is a SimObject, so it takes a generated params struct rather than
 * loose arguments.  Stack-allocating and filling that struct is the same
 * trick src/base/filters/base.test.cc uses; eventq_index must be set
 * because SimObject's constructor reads it.
 */
SSBPParams
testParams()
{
    SSBPParams p;

    p.name = "ssbp.test";
    p.eventq_index = 0;
    p.numEntries = NumEntries;
    p.depCheckShift = DepCheckShift;

    // The values the paper measured, so an unmodified params struct
    // reproduces the golden vectors.
    p.c3Max = 32;
    p.c4Max = 3;
    p.c3Increment = 16;
    p.c3ViolationSet = 15;
    p.c4ForgivenOnC3Zero = false;
    p.ssbd = false;
    // TEMPORARY -- goes with the alwaysBypass knob.
    p.alwaysBypass = false;

    // WARNING: SSBPParams is stack allocated here and its members are
    // not default initialised, so every new parameter must be set
    // above.  Miss one and it picks up whatever was on the stack --
    // a bool reads as true about as often as not, so the symptom is a
    // pile of unrelated golden-vector failures, not a missing field.
    return p;
}

/**
 * Drives one load through the predictor and reports the execution type
 * the PSFP-free model emits.
 *
 * This is the entire reachable state machine: the 2x2 of "did the
 * predictor say wait?" against "did the load actually overlap an older
 * store?".  Types A, C, D and E need a non-zero C0 or C2, both of which
 * belong to PSFP, so they cannot occur here.
 *
 * @param aliasing True if the load overlapped an older store.
 * @return 'B', 'F', 'G' or 'H'.
 */
char
step(SSBP &ssbp, Addr pc, bool aliasing)
{
    if (!ssbp.predictWait(pc)) {
        // The predictor let the load bypass older stores.
        if (!aliasing) {
            // Nothing overlapped, so bypassing was the right call and the
            // FSM sees no input at all.
            return 'H';
        }

        // It overlapped.  The machine squashes and trains type G.
        ssbp.trainViolation(pc);
        return 'G';
    }

    // The load was held back, so a prediction was genuinely exercised and
    // the B/F half of the FSM gets an input.
    ssbp.loadResolved(pc, aliasing);
    return aliasing ? 'B' : 'F';
}

/**
 * Feeds a whole sequence of loads through the predictor.
 *
 * @param inputs One character per load: 'a' aliasing, 'n' non-aliasing.
 * @return One execution type per input, in order.
 */
std::string
run(SSBP &ssbp, Addr pc, const std::string &inputs)
{
    std::string types;

    for (char in : inputs) {
        types += step(ssbp, pc, in == 'a');
    }

    return types;
}

} // anonymous namespace

/**
 * The minimal case from the paper: three aliasing loads saturate C4 and
 * set C3 to 15, then non-aliasing loads decay C3 one step at a time until
 * the load is allowed to bypass again.
 */
TEST(SSBP, MinimalViolationSequence)
{
    SSBP ssbp(testParams());

    EXPECT_EQ(run(ssbp, PcA, "aaa" + std::string(35, 'n')),
              "GGG" + std::string(15, 'F') + std::string(20, 'H'));
}

/**
 * The training prefix from section IV-A.  Every aliasing load here
 * arrives while C3 is still 0, so all three bypass and flush; the third
 * is the one that saturates C4 and finally arms C3.
 */
TEST(SSBP, TrainingPrefix)
{
    SSBP ssbp(testParams());

    const std::string in = std::string(7, 'n') + 'a' + std::string(7, 'n')
        + 'a' + std::string(7, 'n') + 'a';
    const std::string out = std::string(7, 'H') + 'G' + std::string(7, 'H')
        + 'G' + std::string(7, 'H') + 'G';

    EXPECT_EQ(run(ssbp, PcA, in), out);
    EXPECT_EQ(ssbp.getC3(PcA), 15);
    EXPECT_EQ(ssbp.getC4(PcA), 3);
}

/**
 * C4 is incremented before it is compared against its cap.  Get that
 * ordering backwards and the third violation leaves C3 at 0 instead of
 * 15, so this pins it down one event at a time.
 */
TEST(SSBP, C4IsIncrementedBeforeItIsTested)
{
    SSBP ssbp(testParams());

    EXPECT_EQ(run(ssbp, PcA, "a"), "G");
    EXPECT_EQ(ssbp.getC4(PcA), 1);
    EXPECT_EQ(ssbp.getC3(PcA), 0);

    EXPECT_EQ(run(ssbp, PcA, "a"), "G");
    EXPECT_EQ(ssbp.getC4(PcA), 2);
    EXPECT_EQ(ssbp.getC3(PcA), 0);

    // The third violation is the one that arms the predictor.  It still
    // emits G rather than B, because C3 was 0 when the prediction ran.
    EXPECT_EQ(run(ssbp, PcA, "a" + std::string(35, 'n')),
              "G" + std::string(15, 'F') + std::string(20, 'H'));
}

/**
 * Once C3 is non-zero the load waits, so an alias confirms the prediction
 * and reinforces C3 instead of causing a flush.
 */
TEST(SSBP, TypeBOnceTrained)
{
    SSBP ssbp(testParams());

    run(ssbp, PcA, "aaa");
    ASSERT_EQ(ssbp.getC3(PcA), 15);

    EXPECT_EQ(run(ssbp, PcA, "a"), "B");
    EXPECT_EQ(ssbp.getC3(PcA), 31);
}

/** C3 is six bits wide and stops at 32 however often type B fires. */
TEST(SSBP, C3SaturatesAt32)
{
    SSBP ssbp(testParams());

    run(ssbp, PcA, "aaa");

    // 15 -> 31 -> 32 -> 32 -> ...  All of them are type B.
    EXPECT_EQ(run(ssbp, PcA, std::string(10, 'a')), std::string(10, 'B'));
    EXPECT_EQ(ssbp.getC3(PcA), 32);
}

/**
 * C4 saturates at 3 and is never decayed or reset.  That persistence is
 * what makes three violations permanently sticky: a PC that has earned a
 * saturated C4 is re-armed by a single later alias rather than needing
 * three again.
 */
TEST(SSBP, C4SaturatesAndNeverResets)
{
    SSBP ssbp(testParams());

    // violation() is driven directly: once C3 is non-zero the load waits,
    // so no further type G can arise through step().
    for (int i = 0; i < 10; ++i) {
        ssbp.trainViolation(PcA);
    }

    EXPECT_EQ(ssbp.getC4(PcA), 3);
    EXPECT_EQ(ssbp.getC3(PcA), 15);

    // Decay C3 all the way back to zero.
    run(ssbp, PcA, std::string(15, 'n'));
    ASSERT_EQ(ssbp.getC3(PcA), 0);
    EXPECT_EQ(ssbp.getC4(PcA), 3);

    // One alias, not three, brings it straight back.
    EXPECT_EQ(run(ssbp, PcA, "a"), "G");
    EXPECT_EQ(ssbp.getC3(PcA), 15);
}

/**
 * The sequence that cannot be reproduced without PSFP, kept here as the
 * SSBP-only baseline.  Phase 4 must turn each run of four H into four E
 * once C0 exists; if it does not, C0 is wired up wrong.
 */
TEST(SSBP, PsfpFreeBaselineForC0)
{
    SSBP ssbp(testParams());

    const std::string in = std::string("a") + std::string(4, 'n') + 'a'
        + std::string(4, 'n') + 'a' + std::string(16, 'n');
    const std::string out = std::string("G") + std::string(4, 'H') + 'G'
        + std::string(4, 'H') + 'G' + std::string(15, 'F') + 'H';

    EXPECT_EQ(run(ssbp, PcA, in), out);
}

/** Training one PC must not disturb a PC in a different entry. */
TEST(SSBP, EntriesAreIndependent)
{
    SSBP ssbp(testParams());

    run(ssbp, PcA, "aaa");

    EXPECT_EQ(ssbp.getC3(PcA), 15);
    EXPECT_EQ(ssbp.getC3(PcB), 0);
    EXPECT_EQ(ssbp.getC4(PcB), 0);
}

/** clear() is a full wipe of the table, so it discards both counters. */
TEST(SSBP, ClearDiscardsTraining)
{
    SSBP ssbp(testParams());

    run(ssbp, PcA, "aaa");
    ASSERT_EQ(ssbp.getC3(PcA), 15);

    ssbp.clear();

    EXPECT_EQ(ssbp.getC3(PcA), 0);
    EXPECT_EQ(ssbp.getC4(PcA), 0);
}

/**
 * By default C4 is never forgiven, so a PC that has already earned a
 * saturated C4 is re-armed by one later alias rather than three.  This
 * stickiness is the property the paper measured.
 */
TEST(SSBP, C4IsStickyByDefault)
{
    SSBP ssbp(testParams());

    // Arm it, then decay C3 all the way back to zero.
    EXPECT_EQ(run(ssbp, PcA, "aaa" + std::string(15, 'n')),
              "GGG" + std::string(15, 'F'));

    EXPECT_EQ(ssbp.getC3(PcA), 0);
    EXPECT_EQ(ssbp.getC4(PcA), 3);

    // One alias, not three, puts it straight back.
    EXPECT_EQ(run(ssbp, PcA, "a"), "G");
    EXPECT_EQ(ssbp.getC3(PcA), 15);
}

/**
 * With c4ForgivenOnC3Zero the violation history is dropped as soon as C3
 * decays all the way back, so the PC must earn its three violations
 * again.  Deliberately not what the silicon does; it exists to measure
 * what the stickiness costs.
 */
TEST(SSBP, C4IsForgivenWhenC3DecaysIfEnabled)
{
    SSBPParams p = testParams();
    p.c4ForgivenOnC3Zero = true;
    SSBP ssbp(p);

    // Arm it, then decay C3 all the way back to zero.
    EXPECT_EQ(run(ssbp, PcA, "aaa" + std::string(15, 'n')),
              "GGG" + std::string(15, 'F'));

    EXPECT_EQ(ssbp.getC3(PcA), 0);
    EXPECT_EQ(ssbp.getC4(PcA), 0);

    // Under the default policy the next alias would emit G and set C3
    // straight back to 15; here it starts over.
    EXPECT_EQ(run(ssbp, PcA, "a"), "G");
    EXPECT_EQ(ssbp.getC4(PcA), 1);
    EXPECT_EQ(ssbp.getC3(PcA), 0);
}

/**
 * SSBD pins every entry to [Block]: every load waits, whatever the
 * counters say (section VI-A).  Training still happens underneath, so
 * clearing the flag would restore normal behaviour.
 */
TEST(SSBP, SsbdForcesEveryLoadToWait)
{
    SSBPParams p = testParams();
    p.ssbd = true;
    SSBP ssbp(p);

    // An untrained entry would normally bypass, emitting H or G.
    ASSERT_EQ(ssbp.getC3(PcA), 0);

    // Instead every load waits, so only types B and F can occur.
    EXPECT_EQ(run(ssbp, PcA, "nnn"), "FFF");
    EXPECT_EQ(run(ssbp, PcB, "a"), "B");

    // No violation was ever reported, so C4 stayed put.
    EXPECT_EQ(ssbp.getC4(PcA), 0);
}

/** The counter magnitudes are parameters, so a sweep can move them. */
TEST(SSBP, CounterMagnitudesAreConfigurable)
{
    SSBPParams p = testParams();
    p.c3Increment = 4;
    p.c3ViolationSet = 2;
    p.c4Max = 1;
    SSBP ssbp(p);

    // c4Max of 1 means a single violation arms the predictor, and it
    // arms to c3ViolationSet rather than 15.
    EXPECT_EQ(run(ssbp, PcA, "a"), "G");
    EXPECT_EQ(ssbp.getC4(PcA), 1);
    EXPECT_EQ(ssbp.getC3(PcA), 2);

    // Two decay steps, then it bypasses again.
    EXPECT_EQ(run(ssbp, PcA, "nnn"), "FFH");

    // And type B now adds 4 rather than 16.
    run(ssbp, PcA, "a");
    EXPECT_EQ(ssbp.getC3(PcA), 2);
}

/**
 * The reverse-engineered AMD hash, checked against hand-computed values
 * (SS III-C-2, IV-B-1).  At 4096 entries the chunk width is 12, so bit i
 * of the index is A_i ^ A_{i+12} ^ A_{i+24} ^ A_{i+36}.
 *
 * The last two cases matter most: those terms are zero for every real
 * gem5 virtual PC, which is about 23 bits wide, so this is the only
 * place they are exercised until the predictor is fed a physical
 * address.
 */
TEST(SSBP, XorFoldMatchesThePaper)
{
    SSBP ssbp(testParams());

    // Nothing above the low chunk: the fold is the identity.
    ssbp.trainViolation(0x001);
    EXPECT_EQ(ssbp.getC4(0x001), 1);

    // Bit 12 folds onto bit 0, so this lands in the same entry as 0x001
    // and inherits its counter rather than starting fresh.
    EXPECT_EQ(ssbp.getC4(0x1000), 1);

    // Bit 24 folds onto bit 0 as well.
    EXPECT_EQ(ssbp.getC4(0x1000000), 1);

    // And bit 36.  This is the term a 23-bit virtual PC can never reach.
    EXPECT_EQ(ssbp.getC4(0x1000000000ULL), 1);

    // An address differing only above bit 47 is outside the fold, so it
    // shares the entry too.
    EXPECT_EQ(ssbp.getC4(0x1000000000000ULL | 0x001), 1);
}

/**
 * Two distinct PCs sharing a table entry train each other's counters:
 * 0x1000 folds onto 0x001.  This is the property SS V-D fingerprints the
 * predictor with, and the one an attacker relies on -- the victim's load
 * inherits training it never performed.
 */
TEST(SSBP, CollidingPcsShareAnEntry)
{
    SSBP ssbp(testParams());

    // Under mask these two would be different entries (0x001 vs 0x000).
    run(ssbp, 0x001, "aaa");

    EXPECT_EQ(ssbp.getC3(0x1000), 15);
    EXPECT_EQ(ssbp.getC4(0x1000), 3);

    // Already armed, so the victim's first load confirms rather than
    // flushing -- inheritance, which is what the attack needs.
    EXPECT_EQ(run(ssbp, 0x1000, "a"), "B");
}

/**
 * A smaller table folds in smaller chunks rather than truncating the
 * 12-bit hash, so every address bit still reaches the index.
 */
TEST(SSBP, XorFoldChunkWidthTracksTableSize)
{
    SSBPParams p = testParams();
    p.numEntries = 256;           // chunk width 8
    SSBP ssbp(p);

    ssbp.trainViolation(0x001);

    // Bit 8 folds onto bit 0 at this size, where at 4096 it would not.
    EXPECT_EQ(ssbp.getC4(0x100), 1);
}

// TEMPORARY -- verifies the sharedEntry instrumentation, to be removed
// with it.  0x1000 and 0x2003 both fold to index 1: for PC < 2^24 the
// index is (PC & 0xfff) ^ (PC >> 12), so 0^1 == 3^2 == 1.
TEST(SSBP, SharedEntryIsReportedOnCollision)
{
    SSBP ssbp(testParams());

    // First PC to touch the entry: nothing to share with yet.
    EXPECT_FALSE(ssbp.predict(0x1000, true).sharedEntry);
    EXPECT_FALSE(ssbp.predict(0x1000, true).sharedEntry);

    // A second, colliding PC.  Its own access is the one that proves
    // the collision, so it must report true immediately.
    EXPECT_TRUE(ssbp.predict(0x2003, true).sharedEntry);
    EXPECT_TRUE(ssbp.predict(0x1000, true).sharedEntry);

    // A PC landing on an untouched entry must stay clean.
    EXPECT_FALSE(ssbp.predict(0x4000, true).sharedEntry);

    // Stores never consult the table at all.
    EXPECT_FALSE(ssbp.predict(0x1000, false).sharedEntry);
}
