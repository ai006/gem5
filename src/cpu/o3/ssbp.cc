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

SSBP::SSBP(std::string_view name_, int numEntries_)
    : Named(name_),
      numEntries(numEntries_)
{
    DPRINTF(SSBP, "SSBP: Creating SSBP object.\n");
    if (!isPowerOf2(numEntries)) {
        fatal("SSBP: number of entries must be a power of 2!\n");
    }
    ssbpEntries.resize(numEntries);

}

SSBP::~SSBP()
{
}

void
SSBP::init(int numEntries_)
{
    if (!isPowerOf2(numEntries_)) {
        fatal("SSBP: number of entries must be a power of 2!\n");
    }

    numEntries = numEntries_;
    ssbpEntries.resize(numEntries);
}

bool
SSBP::predictWait(Addr load_PC) const
{
    const SSBPEntry &entry = ssbpEntries[getIndex(load_PC)];
    return (entry.getC3() > 0);
}

void
SSBP::violation(Addr load_PC)
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
    }

    DPRINTF(SSBP, "Load resolved at PC %#x: type %c, C3 %i, C4 %i\n",
            load_PC, aliased ? 'B' : 'F', entry.getC3(), entry.getC4());
}

void
SSBP::clear()
{
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

InstSeqNum
SSBP::checkInst(Addr PC)
{
    bool prediction = predictWait(PC);
    InstSeqNum dep = prediction ? youngestStore.SeqNum : 0;

    DPRINTF(SSBP, "Check PC %#x: C3 %i, predict %s, producer [sn:%lli]\n",
            PC, getC3(PC), prediction ? "wait" : "go", dep);

    return dep;
}


void
SSBP::insertStore(Addr store_PC, InstSeqNum store_seq_num,
                         ThreadID tid)
{
    youngestStore.store_PC = store_PC;
    youngestStore.SeqNum = store_seq_num;
    youngestStore.tid = tid;
}

void
SSBP::squash(InstSeqNum squashed_num, ThreadID tid)
{
    if (youngestStore.SeqNum > squashed_num) {
        youngestStore.SeqNum = 0;
    }
}

} // namespace o3
} // namespace gem5
