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
#include <unordered_map>
#include <vector>

#include "base/intmath.hh"
#include "base/named.hh"
#include "base/types.hh"
#include "cpu/inst_seq.hh"
#include "cpu/o3/mem_dep_pred.hh"
#include "params/SSBP.hh"


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
class SSBP : public MemDepPredictor
{
    public:
        typedef SSBPParams Params;

        /** Creates the predictor with the configured table size. */
        SSBP(const Params &p);

        /** Default destructor. */
        ~SSBP();

        /** Predicts whether a load should wait for older in-flight stores.
         *  Keyed on the load PC alone; the predictor never names a specific
         *  store.  The full AMD rule is (C0 > 0 || C3 > 0), where C0 belongs
         *  to the predictive store forwarding predictor.  PSFP is not
         *  modelled here, so C0 is permanently zero and the rule reduces to
         *  C3 > 0.
         *  @return True if the load should be held back.
         */
        bool predictWait(Addr load_PC) const;

        /** Decides whether this load should wait.  SSBP is a Kind B
         *  predictor: it never names a store, so a wait comes back as
         *  YoungestOlderStore and MemDepUnit resolves which store that is.
         *  Stores are not predicted at all -- the counters are trained by
         *  and for loads.
         */
        MemDepPrediction predict(Addr pc, bool is_load) override;

        /** Records a memory ordering violation: a load bypassed an older
         *  store to the same address and had to be squashed.  Type G of the
         *  paper's state machine.  C4 counts these events and saturates;
         *  C3 stays at zero until C4 has saturated, so a load must cause
         *  several violations before it is held back for any length of time.
         *
         *  store_PC is accepted to match the common predictor interface but
         *  deliberately unused: Table II shows C3 and C4 are selected by the
         *  load address alone.  PSFP is the half that needs the pair.
         */
        void violation(Addr store_PC, Addr load_PC) override;

        /** The type G update itself, keyed on the load alone.  Kept
         *  separate from the interface method above so that the state
         *  machine stays drivable with no CPU and no store PC, which is
         *  what the golden-vector unit tests need.  Do not make this an
         *  overload of violation(): a one-argument overload alongside the
         *  two-argument override binds wrongly through a base pointer.
         */
        void trainViolation(Addr load_PC);

        /** Trains the predictor when a load that was held back resolves.
         *  Covers types B and F: a load that overlapped an older store
         *  confirms the prediction and raises C3 sharply, while one that
         *  overlapped nothing decays C3 by a single step, so a run of
         *  harmless executions eventually re-enables bypassing.
         *  @param aliased True if the load overlapped an older store.
         */
        void loadResolved(Addr load_PC, bool aliased);

        /** Resets every entry, discarding all training. */
        void clear() override;

        /** Counter inspection.  Not used by the dependence unit; these exist
         *  for statistics, unit tests, and for sampling the distribution of
         *  C3 across the table.
         */
        uint8_t getC3(Addr load_PC) const;
        uint8_t getC4(Addr load_PC) const;

    private:
        /** Turns a load address into a table index, using the
         *  reverse-engineered AMD hash (SS III-C-2, IV-B-1): the address
         *  is folded on itself in index-sized chunks and XORed, so bits
         *  from across the whole address decide the entry rather than
         *  only the lowest ones.  At the default 4096 entries the chunk
         *  width is 12 and this is exactly the paper's
         *  h_i = A_i ^ A_{i+12} ^ A_{i+24} ^ A_{i+36}.
         *
         *  Which loads collide is the whole point: two loads sharing an
         *  entry train each other's counters, which is what lets one
         *  address space train another's predictor.
         */
        unsigned getIndex(Addr load_PC) const;

        /** Width of one fold chunk, log2(numEntries).  12 at the default
         *  table size, which is what makes the fold paper-exact there and
         *  lets a smaller table generalise instead of truncating. */
        unsigned indexBits = 0;

        /** Widest address the fold covers.  x86-64 virtual addresses are
         *  48 bits and physical addresses no wider, so folding past this
         *  would only mix in zeroes.  At indexBits 12 it gives exactly
         *  the paper's four terms over bits 0..47. */
        static constexpr unsigned AddrBits = 48;

    private:
        int numEntries = 0;
        unsigned depCheckShift = 0;

        class SSBPEntry
        {
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

        /** Counter limits and update magnitudes, defaulting to the values
         *  the paper measured (Table I, Table IV).  Parameters rather than
         *  constants so each can be swept from the configuration.
         */
        uint8_t C3Max = 32;
        uint8_t C4Max = 3;
        uint8_t C3Violation = 15;
        uint8_t C3Increment = 16;

        /** Whether C4 is forgiven when C3 decays back to zero.  The paper
         *  never observed C4 reset, which is what makes three violations
         *  permanently sticky, so this is off by default.
         */
        bool c4ForgivenOnC3Zero = false;

        /** Speculative Store Bypass Disable.  Pins every entry to [Block]
         *  so no load is allowed to bypass an unresolved store.
         */
        bool ssbd = false;

        /** Index by the instruction's physical address.  See
         *  wantsPhysicalIndex(). */
        bool useIPA = false;

        struct DelayedLoad
        {
            Addr loadPC = 0;
            Addr storePC = 0;
            Addr storeAddr = 0;      // filled in when the store wakes the load
            unsigned storeSize = 0;  // 0 means the store never resolved
        };

        std::unordered_map<InstSeqNum, DelayedLoad> delayedLoads;



    //The functions listed here are for making the SSBP predictor
    //work with the current
    //system of lsq. Non are mentioned in the paper
    public:
        /** Drops the tracked store if it was squashed.  The C3 and C4
         *  counters are deliberately left alone: real SSBP is not rolled
         *  back on a squash, which is the behaviour the transient execution
         *  attacks rely on.
         */
        void squash(InstSeqNum squashed_num, ThreadID tid) override
        {}

        /** Notification that a memory op has issued.  StoreSet retires its
         *  last-fetched-store entry here; SSBP keeps no such table, so this
         *  is intentionally empty.
         */
        void
        issued(Addr issued_PC, InstSeqNum issued_seq_num,
               bool is_store) override
        {}

        /** Records that SSBP held this load back, and on which store.  The
         *  address fields stay empty until the store wakes it. */
        void noteDelayedLoad(InstSeqNum sn, Addr load_PC,
                             Addr store_PC) override;

        /** Records the address range of the store a delayed load was waiting
         *  on, captured at the moment that store wakes it. */
        void noteProducerAddr(InstSeqNum sn, Addr store_addr,
                              unsigned store_size) override;

        /** A load has executed.  If SSBP parked it, decide whether it
        *  overlapped the store it waited on, apply the type B / F update,
        *  and drop the record. */
        MemDepTraining loadExecuted(InstSeqNum sn, Addr load_addr,
                                    unsigned load_size) override;

        /** Drops a record without training it, for squashed loads. */
        void forgetDelayedLoad(InstSeqNum sn) override;

        /** Decides whether an executed load overlapped the store it
        *  was made to wait on.  Comparison is done at LSQDepCheckShift
        *  granularity, so it agrees with LSQUnit::checkViolations rather
        *  than being byte-exact — 16 bytes by default, which reports
        *  aliasing more often than the silicon the paper's thresholds
        *  came from.
        *  @param rec The delayed-load record, holding the store's range.
        *  @param load_addr The load's effective address.
        *  @param load_size The load's access size in bytes.
        *  @return True if the ranges overlap.  False if the store never
        *  resolved an address, since there is then nothing to compare against.
        */
        bool overlaps(const DelayedLoad &rec, Addr load_addr,
               unsigned load_size) const;

        bool drained() const override { return delayedLoads.empty(); }

        /** SSBP is selected by the load's physical address on real
         *  hardware, which is why it leaks across processes.  Off by
         *  default until the fold hash flips with it. */
        bool wantsPhysicalIndex() const override { return useIPA; }
};


} // namespace o3
} // namespace gem5


#endif // __CPU_O3_SSBP_HH__
