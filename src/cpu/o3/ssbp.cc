/*
 * Copyright (c) 2004-2006 The Regents of The University of Michigan
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

#include "cpu/o3/ssbp.hh"

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/SSBP.hh"

namespace gem5
{

namespace o3
{

void
SSBP::SSBPEntry::reset()
{
    setC3(0);
    setC4(0);
}

SSBP::SSBP(const Params &p)
    : MemDepPredictor(p),
      numEntries(p.numEntries),
      depCheckShift(p.depCheckShift),
      C3Max(p.c3Max),
      C4Max(p.c4Max),
      C3Violation(p.c3ViolationSet),
      C3Increment(p.c3Increment),
      c4ForgivenOnC3Zero(p.c4ForgivenOnC3Zero),
      ssbd(p.ssbd)
{
    DPRINTF(SSBP, "SSBP: Creating SSBP object.\n");

    if (!isPowerOf2(numEntries)) {
        fatal("SSBP: number of entries must be a power of 2!\n");
    }

    // The counters are 6 and 2 bits wide in the silicon (Table IV).  A
    // sweep that exceeds those widths is no longer modelling the same
    // structure, so say so rather than quietly simulating a wider table.
    if (C3Max > 63) {
        fatal("SSBP: c3Max is %i but C3 is 6 bits wide (max 63).\n", C3Max);
    }

    if (C4Max > 3) {
        fatal("SSBP: c4Max is %i but C4 is 2 bits wide (max 3).\n", C4Max);
    }

    if (C3Violation > C3Max) {
        fatal("SSBP: c3ViolationSet (%i) exceeds c3Max (%i).\n",
              C3Violation, C3Max);
    }

    if (ssbd) {
        DPRINTF(SSBP, "SSBP: SSBD set; every load will be held back.\n");
    }

    ssbpEntries.resize(numEntries);
}

SSBP::~SSBP()
{
}


bool
SSBP::predictWait(Addr load_PC) const
{
    // SSBD pins every entry to [Block], so the predictor always reports
    // aliasing and no load is allowed to bypass (section VI-A).  The
    // counters keep training underneath; only the answer is forced.
    if (ssbd)
        return true;

    const SSBPEntry &entry = ssbpEntries[getIndex(load_PC)];
    return (entry.getC3() > 0);
}

MemDepPrediction
SSBP::predict(Addr pc, bool is_load)
{
    // Kind B, and load side only.  Stores are never held back by SSBP;
    // the counters are trained by and for loads.
    if (!is_load || !predictWait(pc))
        return {};

    // SSBP names no store, so MemDepUnit has to resolve which one.
    return {MemDepPrediction::YoungestOlderStore, 0};
}

void
SSBP::violation(Addr store_PC, Addr load_PC)
{
    // store_PC is deliberately ignored: Table II shows C3 and C4 are
    // selected by the load address alone.
    trainViolation(load_PC);
}

void
SSBP::trainViolation(Addr load_PC)
{
    SSBPEntry &entry = ssbpEntries[getIndex(load_PC)];

    uint8_t c4 = entry.getC4();
    if (c4 < C4Max) {
        c4++;
        entry.setC4(c4);
    }

    entry.setC3((c4 >= C4Max) ? C3Violation : 0);

    DPRINTF(SSBP, "Violation at PC %#x: C3 %i, C4 %i\n",
            load_PC, entry.getC3(), entry.getC4());
}

void
SSBP::loadResolved(Addr load_PC, bool aliased)
{
    SSBPEntry &entry = ssbpEntries[getIndex(load_PC)];
    uint8_t c3 = entry.getC3();

    if (aliased) {
        // Type B: the wait was justified.
        unsigned next = c3 + C3Increment;
        entry.setC3((next > C3Max) ? C3Max : next);
    } else if (c3 > 0) {
        // Type F: the wait was needless, decay one step.
        entry.setC3(c3 - 1);

        // Optional forgiveness: once C3 has decayed all the way back, let
        // C4 go with it, so the PC must earn its violations again rather
        // than being re-armed by a single later alias.  Deliberately not
        // what the silicon does; it exists to measure what the stickiness
        // costs.
        if (c4ForgivenOnC3Zero && entry.getC3() == 0)
            entry.setC4(0);
    }

    DPRINTF(SSBP, "Load resolved at PC %#x: type %c, C3 %i, C4 %i\n",
            load_PC, aliased ? 'B' : 'F', entry.getC3(), entry.getC4());
}

void
SSBP::clear()
{
    // A full wipe of the table, as on a CPU switchover.  This models the
    // structure being torn down rather than an FSM transition, so it
    // discards C4 as well: the paper's "C4 never resets" is a statement
    // about Table I, not about the predictor surviving a reset.
    for (SSBPEntry &entry : ssbpEntries) {
        entry.reset();
    }
}

uint8_t
SSBP::getC3(Addr load_PC) const
{
    return ssbpEntries[getIndex(load_PC)].getC3();
}

uint8_t
SSBP::getC4(Addr load_PC) const
{
    return ssbpEntries[getIndex(load_PC)].getC4();
}

void
SSBP::noteDelayedLoad(InstSeqNum sn, Addr load_PC, Addr store_PC)
{
    DelayedLoad &rec = delayedLoads[sn];

    rec.loadPC = load_PC;
    rec.storePC = store_PC;
    rec.storeAddr = 0;
    rec.storeSize = 0;

    DPRINTF(SSBP, "Delayed load PC %#x [sn:%lli] behind store PC %#x\n",
            load_PC, sn, store_PC);
}

/** Records the address range of the store a delayed load was waiting
*  on, captured at the moment that store wakes it. */
void
SSBP::noteProducerAddr(InstSeqNum sn, Addr store_addr,
                        unsigned store_size)
{
    auto it = delayedLoads.find(sn);

    // A store wakes every load queued behind it.  Only the ones SSBP
    // parked are tracked, so a miss here is the common case.
    if (it == delayedLoads.end())
        return;

    it->second.storeAddr = store_addr;
    it->second.storeSize = store_size;
}

/** A load has executed.  If SSBP parked it, decide whether it
*  overlapped the store it waited on, apply the type B / F update,
*  and drop the record. */
void
SSBP::loadExecuted(InstSeqNum sn, Addr load_addr, unsigned load_size)
{
    auto it = delayedLoads.find(sn);

    // SSBP never parked this load, so no prediction was exercised and
    // the FSM should see no input at all.
    if (it == delayedLoads.end())
        return;

    const DelayedLoad &rec = it->second;
    bool aliased = overlaps(rec, load_addr, load_size);

    loadResolved(rec.loadPC, aliased);

    // Dropping the record here is what stops a load that re-executes
    // from training twice.
    delayedLoads.erase(it);
}

/** Drops a record without training it, for squashed loads. */
void
SSBP::forgetDelayedLoad(InstSeqNum sn)
{
    delayedLoads.erase(sn);
}

bool
SSBP::overlaps(const DelayedLoad &rec, Addr load_addr,
               unsigned load_size) const
{
    // storeSize 0 means the producer never resolved an address.
    if (rec.storeSize == 0)
        return false;

    Addr ld_lo = load_addr >> depCheckShift;
    Addr ld_hi = (load_addr + load_size - 1) >> depCheckShift;
    Addr st_lo = rec.storeAddr >> depCheckShift;
    Addr st_hi = (rec.storeAddr + rec.storeSize - 1) >> depCheckShift;

    return st_hi >= ld_lo && st_lo <= ld_hi;
}

} // namespace o3
} // namespace gem5
