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

#ifndef __CPU_O3_SSBP_HH__
#define __CPU_O3_SSBP_HH__

#include <assert.h>
#include <cstdint>
#include <string_view>
#include <vector>

#include "base/intmath.hh"
#include "base/named.hh"
#include "base/types.hh"
#include "cpu/inst_seq.hh"


namespace gem5
{

namespace o3
{


/**
 * Implements the reverse engineered AMD's Speculative-Store-Bypass Predictor
 * found on AMD Zen 3 CPU for determining if memory
 * instructions are dependent upon each other.  See paper "Uncovering
 * and Exploiting AMD Speculative Memory Access Predictors for Fun and
 * Profit" by Lyu and Qiu.
 * modeled according to the paper by Ange and Rafi
 */
class SSBP : public Named
{
    public:
        /** Default constructor.  init() must be called prior to use. */
        SSBP() : Named("SSBP") {};
        /** Creates store set predictor with given table sizes. */
        SSBP(std::string_view name_, int numEntries_);
        /** Default destructor. */
        ~SSBP();

        /** Initializes the store set predictor with the given table sizes. */
        void init(int numEntries_);

        /** Predicts whether a load should wait for older in-flight stores.
         *  Keyed on the load PC alone; the predictor never names a specific
         *  store.  The full AMD rule is (C0 > 0 || C3 > 0), where C0 belongs
         *  to the predictive store forwarding predictor.  PSFP is not
         *  modelled here, so C0 is permanently zero and the rule reduces to
         *  C3 > 0.
         *  @return True if the load should be held back.
         */
        bool predictWait(Addr load_PC) const;

        /** Records a memory ordering violation: a load bypassed an older
         *  store to the same address and had to be squashed.  Type G of the
         *  paper's state machine.  C4 counts these events and saturates;
         *  C3 stays at zero until C4 has saturated, so a load must cause
         *  several violations before it is held back for any length of time.
         */
        void violation(Addr load_PC);

        /** Trains the predictor when a load that was held back resolves.
         *  Covers types B and F: a load that overlapped an older store
         *  confirms the prediction and raises C3 sharply, while one that
         *  overlapped nothing decays C3 by a single step, so a run of
         *  harmless executions eventually re-enables bypassing.
         *  @param aliased True if the load overlapped an older store.
         */
        void loadResolved(Addr load_PC, bool aliased);

        /** Resets every entry, discarding all training. */
        void clear();

        /** Counter inspection.  Not used by the dependence unit; these exist
         *  for statistics, unit tests, and for sampling the distribution of
         *  C3 across the table.
         */
        uint8_t getC3(Addr load_PC) const;
        uint8_t getC4(Addr load_PC) const;

        /** Checks if load isntruction with the given PC is dependent upon
        * any store.  @return Returns the sequence number of the store
        * instruction this PC is dependent upon.  Returns 0 if none.
        */
        InstSeqNum checkInst(Addr PC);

    private:
        inline unsigned getIndex(Addr load_PC) const
        { return load_PC & (numEntries - 1); }


    private:
        int numEntries = 0;
        class SSBPEntry{
            private:
                uint8_t c3;
                uint8_t c4;

            public:
                SSBPEntry() : c3(0), c4(0) { }
                void setC3(uint8_t _c3) { c3 = _c3; }
                void setC4(uint8_t _c4) { c4 = _c4; }
                uint8_t getC3() const { return c3; }
                uint8_t getC4() const { return c4; }
                void reset();

        };
        std::vector<SSBPEntry> ssbpEntries;
        /** Counter limits and update magnitudes (Table I, Table IV) of paper. */
        static constexpr uint8_t C3Max = 32;
        static constexpr uint8_t C4Max = 3;
        static constexpr uint8_t C3Violation = 15;
        static constexpr uint8_t C3Increment = 16;

        /** The most recently dispatched store.  SSBP predicts only whether a
         *  load should wait, never which store to wait for, so a predicted
         *  wait is attached to the youngest store in flight at the time the
         *  load is dispatched.  Updated by insertStore(), rewound by
         *  squash().  A sequence number of zero means nothing is being
         *  tracked, matching the "no dependency" value StoreSet returns.
         *  The store PC is unused today; PSFP needs it, as that predictor is
         *  indexed by the store and load addresses together.
         */
         struct {
            Addr store_PC = 0;
            InstSeqNum SeqNum = 0;
            ThreadID tid = InvalidThreadID;
         } youngestStore;

    public:
        /** Records a store as it enters the scheduler.  StoreSet uses this
         *  to maintain its last-fetched-store table; SSBP uses it to track
         *  the youngest store in flight, which is the store a predicted
         *  wait gets attached to.  Called from both MemDepUnit::insert and
         *  MemDepUnit::insertNonSpec, so non-speculative stores are seen
         *  as well.
         */
        void insertStore(Addr store_PC, InstSeqNum store_seq_num,
                         ThreadID tid);

        /** Drops the tracked store if it was squashed.  The C3 and C4
         *  counters are deliberately left alone: real SSBP is not rolled
         *  back on a squash, which is the behaviour the transient execution
         *  attacks rely on.
         */
        void squash(InstSeqNum squashed_num, ThreadID tid);

        /** Notification that a memory op has issued.  StoreSet retires its
         *  last-fetched-store entry here; SSBP keeps no such table, so this
         *  is intentionally empty.
         */
        void
        issued(Addr issued_PC, InstSeqNum issued_seq_num, bool is_store)
        {}



};


} // namespace o3
} // namespace gem5


#endif // __CPU_O3_SSBP_HH__
