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

#ifndef __CPU_O3_MEM_DEP_PRED_HH__
#define __CPU_O3_MEM_DEP_PRED_HH__

#include "base/types.hh"
#include "cpu/inst_seq.hh"
#include "params/MemDepPredictor.hh"
#include "sim/sim_object.hh"

namespace gem5
{

namespace o3
{

/**
 * What a predictor decides about one memory operation.
 *
 * The three cases are exactly the Kind A / Kind B distinction from the
 * memory dependence prediction literature.  A Kind A predictor (store
 * sets) names the store it wants the op to wait for.  A Kind B predictor
 * (SSBP) answers only "should this wait?" and leaves the choice of store
 * to MemDepUnit, which is the only place that knows which stores are
 * still in flight.
 */
struct MemDepPrediction
{
    enum Kind
    {
        /** Issue as soon as the registers are ready. */
        NoDependence,
        /** Wait for the specific store named by producer. */
        NamedStore,
        /** Wait, but let MemDepUnit pick the store. */
        YoungestOlderStore,
    };

    Kind kind = NoDependence;

    /** Only meaningful when kind is NamedStore. */
    InstSeqNum producer = 0;

    /** TEMPORARY INSTRUMENTATION -- not part of the prediction.  True
     *  when the table entry consulted had already been used by a
     *  different PC, i.e. the index hash collided.  Nothing acts on it;
     *  MemDepUnit only counts it.  Left false by any predictor that does
     *  not track this.  Delete with sharedEntryPredictions; see the
     *  Cleanup section of IMPLEMENTATION_CHECKLIST.md.
     */
    bool sharedEntry = false;
};

/**
 * What a predictor learned when a load it held back finally executed.
 *
 * Reported back to MemDepUnit so the unit can count outcomes without
 * knowing which predictor produced them, and so that the predictors
 * themselves stay free of the statistics runtime -- a registered stat in
 * a predictor's translation unit pulls statistics.cc, and from there
 * Root, into any unit test that links it.
 */
enum class MemDepTraining
{
    /** The predictor never held this load back, so no prediction was
     *  exercised and its state machine saw no input. */
    None,
    /** The load did overlap the store it waited for: the wait was
     *  justified.  Type B of the paper's taxonomy. */
    Confirmed,
    /** The load overlapped nothing: the stall bought nothing.  Type F. */
    Needless,
};

/**
 * Interface every memory dependence predictor implements.
 *
 * One instance per hardware thread, created by the configuration and
 * handed to that thread's MemDepUnit.  Per-thread duplication is
 * deliberate: the AMD predictors this models are partitioned amongst SMT
 * threads, so sharing one instance across threads would be wrong rather
 * than merely different.
 *
 * Methods that only some predictors need have empty defaults rather than
 * being pure virtual, so MemDepUnit can call them unconditionally instead
 * of branching on which predictor is configured.
 */
class MemDepPredictor : public SimObject
{
  public:
    typedef MemDepPredictorParams Params;

    MemDepPredictor(const Params &p) : SimObject(p) {}

    virtual ~MemDepPredictor() = default;

    /**
     * Decides whether this memory operation must wait, and on what.
     *
     * @param pc The operation's PC.
     * @param is_load True for loads; predictors that only track loads
     * gate on this, while store_set deliberately builds store to store
     * edges as well.
     */
    virtual MemDepPrediction predict(Addr pc, bool is_load) = 0;

    /**
     * Records that a load illegally passed an older store to the same
     * address and had to be squashed.
     *
     * Carries both PCs so that one signature serves every predictor:
     * store_set needs the pair, while SSBP ignores the store because
     * its counters are selected by the load address alone.
     */
    virtual void violation(Addr store_PC, Addr load_PC) = 0;

    /** Discards all training. */
    virtual void clear() = 0;

    /** Records a store as it enters the scheduler. */
    virtual void
    insertStore(Addr store_PC, InstSeqNum store_seq_num, ThreadID tid)
    {}

    /** Records a memory operation as issued. */
    virtual void
    issued(Addr issued_PC, InstSeqNum issued_seq_num, bool is_store)
    {}

    /** Drops any per-instruction tracking newer than the squash point. */
    virtual void squash(InstSeqNum squashed_num, ThreadID tid) {}

    /**
     * Notes that the predictor held this load back, and on which store.
     * Only predictors that train on whether the wait was justified need
     * this; the rest ignore it.
     */
    virtual void
    noteDelayedLoad(InstSeqNum sn, Addr load_PC, Addr store_PC)
    {}

    /** Notes the address range of the store a delayed load waited on. */
    virtual void
    noteProducerAddr(InstSeqNum sn, Addr store_addr, unsigned store_size)
    {}

    /** Notes that a load has executed, so any wait can now be judged.
     *  @return What the predictor trained, if anything. */
    virtual MemDepTraining
    loadExecuted(InstSeqNum sn, Addr load_addr, unsigned load_size)
    {
        return MemDepTraining::None;
    }

    /** Drops a delayed-load record without training it. */
    virtual void forgetDelayedLoad(InstSeqNum sn) {}

    /** False while the predictor still holds in-flight tracking. */
    virtual bool drained() const { return true; }

    /**
     * Whether this predictor wants to be indexed by the load
     * instruction's physical address rather than its virtual one.
     *
     * Asked rather than assumed so that MemDepUnit, which is the only
     * place that can translate, does not need to know which predictor it
     * is holding.  Translation is not free, so a predictor that does not
     * care keeps the default and pays nothing.
     */
    virtual bool wantsPhysicalIndex() const { return false; }
};

} // namespace o3
} // namespace gem5

#endif // __CPU_O3_MEM_DEP_PRED_HH__
