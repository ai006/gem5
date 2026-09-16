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
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

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
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

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
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

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
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

    run(ssbp, PcA, "aaa");
    ASSERT_EQ(ssbp.getC3(PcA), 15);

    EXPECT_EQ(run(ssbp, PcA, "a"), "B");
    EXPECT_EQ(ssbp.getC3(PcA), 31);
}

/** C3 is six bits wide and stops at 32 however often type B fires. */
TEST(SSBP, C3SaturatesAt32)
{
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

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
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

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
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

    const std::string in = std::string("a") + std::string(4, 'n') + 'a'
        + std::string(4, 'n') + 'a' + std::string(16, 'n');
    const std::string out = std::string("G") + std::string(4, 'H') + 'G'
        + std::string(4, 'H') + 'G' + std::string(15, 'F') + 'H';

    EXPECT_EQ(run(ssbp, PcA, in), out);
}

/** Training one PC must not disturb a PC in a different entry. */
TEST(SSBP, EntriesAreIndependent)
{
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

    run(ssbp, PcA, "aaa");

    EXPECT_EQ(ssbp.getC3(PcA), 15);
    EXPECT_EQ(ssbp.getC3(PcB), 0);
    EXPECT_EQ(ssbp.getC4(PcB), 0);
}

/**
 * Two PCs sharing their low 12 bits share an entry and therefore train
 * each other.  That is not a defect: it is the index aliasing the paper
 * exploits, and section V-D fingerprints the predictor with it.  Pinned
 * here so Phase 3 has to make a deliberate decision when it replaces the
 * mask with the reverse-engineered 12-bit XOR hash, rather than changing
 * this behaviour by accident.
 */
TEST(SSBP, CollidingPcsShareAnEntry)
{
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

    run(ssbp, PcA, "aaa");

    EXPECT_EQ(ssbp.getC3(PcCollidesWithA), 15);
    EXPECT_EQ(ssbp.getC4(PcCollidesWithA), 3);

    // The collision is total, so the victim PC is already armed: its
    // very first load waits and confirms instead of flushing.
    EXPECT_EQ(run(ssbp, PcCollidesWithA, "a"), "B");
}

/** clear() discards training, including the sticky C4. */
TEST(SSBP, ClearDiscardsTraining)
{
    SSBP ssbp("ssbp.test", NumEntries, DepCheckShift);

    run(ssbp, PcA, "aaa");
    ASSERT_EQ(ssbp.getC3(PcA), 15);

    ssbp.clear();

    EXPECT_EQ(ssbp.getC3(PcA), 0);
    EXPECT_EQ(ssbp.getC4(PcA), 0);
}
