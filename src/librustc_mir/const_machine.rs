use rustc_middle::mir;
use rustc_middle::ty::{self, Ty};
use std::borrow::Cow;
use rustc_data_structures::fx::FxHashMap;

use rustc_hir::def_id::DefId;
use rustc_middle::mir::{AssertMessage, Local};

use crate::interpret::{
    self, AllocId, Allocation, Frame, ImmTy, InterpCx, InterpResult, Memory, MemoryKind,
    OpTy, PlaceTy, Pointer, Scalar, Operand as InterpOperand
};

pub trait ConstMachine<'mir, 'tcx>: Sized {
    type MemoryExtra;
    // FIXME: Can't figure out how to not have this assoc. type
    // here. Because const_prop and const_eval machines set this to !,
    // the compiler gives an `unreachable expression` if I set the
    // type to `!` for ConstMachine
    type ExtraFnVal: ::std::fmt::Debug + Copy;

    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, ()>],
        ret: Option<(PlaceTy<'tcx, ()>, mir::BasicBlock)>,
        _unwind: Option<mir::BasicBlock>, // unwinding is not supported in consts
    ) -> InterpResult<'tcx, Option<&'mir mir::Body<'tcx>>>;

    fn call_extra_fn(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        fn_val: Self::ExtraFnVal,
        _args: &[OpTy<'tcx, ()>],
        _ret: Option<(PlaceTy<'tcx, ()>, mir::BasicBlock)>,
        _unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx>;

    fn call_intrinsic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, ()>],
        ret: Option<(PlaceTy<'tcx, ()>, mir::BasicBlock)>,
        _unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx>;

    fn assert_panic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        msg: &AssertMessage<'tcx>,
        _unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx>;

    fn ptr_to_int(
        _mem: &Memory<'mir, 'tcx, Self>,
        _ptr: Pointer<()>,
    ) -> InterpResult<'tcx, u64>;

    fn binary_ptr_op(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        _bin_op: mir::BinOp,
        _left: ImmTy<'tcx, ()>,
        _right: ImmTy<'tcx, ()>,
    ) -> InterpResult<'tcx, (Scalar<()>, bool, Ty<'tcx>)>;

    fn init_allocation_extra<'b>(
        _memory_extra: &Self::MemoryExtra,
        _id: AllocId,
        alloc: Cow<'b, Allocation>,
        _kind: Option<MemoryKind<!>>,
    ) -> (Cow<'b, Allocation<(), ()>>, ());

    fn tag_global_base_pointer(_memory_extra: &Self::MemoryExtra, _id: AllocId)
        -> ();

    fn box_alloc(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        _dest: PlaceTy<'tcx, ()>,
    ) -> InterpResult<'tcx>;

    fn before_terminator(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }

    #[inline(always)]
    fn stack_push(_ecx: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }

    fn before_access_global(
        memory_extra: &Self::MemoryExtra,
        alloc_id: AllocId,
        allocation: &Allocation,
        static_def_id: Option<DefId>,
        is_write: bool,
    ) -> InterpResult<'tcx>;

    fn init_frame_extra(
        _ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx, ()>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx, (), ()>>;

    fn access_local(
        _ecx: &InterpCx<'mir, 'tcx, Self>,
        frame: &Frame<'mir, 'tcx, (), ()>,
        local: Local,
    ) -> InterpResult<'tcx, InterpOperand<()>> {
        frame.locals[local].access()
    }
}

impl<'mir, 'tcx, T: ConstMachine<'mir, 'tcx>> interpret::Machine<'mir, 'tcx> for T {
    type MemoryKind = !;
    type PointerTag = ();
    type ExtraFnVal = T::ExtraFnVal;

    type FrameExtra = ();
    type MemoryExtra = T::MemoryExtra;
    type AllocExtra = ();

    type MemoryMap = FxHashMap<AllocId, (MemoryKind<!>, Allocation)>;

    const GLOBAL_KIND: Option<Self::MemoryKind> = None; // no copying of globals from `tcx` to machine memory

    #[inline(always)]
    fn enforce_alignment(_memory_extra: &Self::MemoryExtra) -> bool {
        // We do not check for alignment to avoid having to carry an `Align`
        // in `ConstValue::ByRef`.
        false
    }

    #[inline(always)]
    fn enforce_validity(_ecx: &InterpCx<'mir, 'tcx, Self>) -> bool {
        false // for now, we don't enforce validity
    }

    fn find_mir_or_eval_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Self::PointerTag>],
        ret: Option<(PlaceTy<'tcx, Self::PointerTag>, mir::BasicBlock)>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir mir::Body<'tcx>>> {
        ConstMachine::find_mir_or_eval_fn(ecx, instance, args, ret, unwind)
    }

    fn call_extra_fn(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        fn_val: Self::ExtraFnVal,
        args: &[OpTy<'tcx, Self::PointerTag>],
        ret: Option<(PlaceTy<'tcx, Self::PointerTag>, mir::BasicBlock)>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        ConstMachine::call_extra_fn(ecx, fn_val, args, ret, unwind)
    }

    fn call_intrinsic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Self::PointerTag>],
        ret: Option<(PlaceTy<'tcx, Self::PointerTag>, mir::BasicBlock)>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        ConstMachine::call_intrinsic(ecx, instance, args, ret, unwind)
    }

    fn assert_panic(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        msg: &AssertMessage<'tcx>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        ConstMachine::assert_panic(ecx, msg, unwind)
    }

    fn binary_ptr_op(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Self::PointerTag>,
        right: ImmTy<'tcx, Self::PointerTag>,
    ) -> InterpResult<'tcx, (Scalar<Self::PointerTag>, bool, Ty<'tcx>)> {
        ConstMachine::binary_ptr_op(ecx, bin_op, left, right)
    }

    fn box_alloc(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        dest: PlaceTy<'tcx, Self::PointerTag>,
    ) -> InterpResult<'tcx> {
        ConstMachine::box_alloc(ecx, dest)
    }

    fn init_allocation_extra<'b>(
        memory_extra: &Self::MemoryExtra,
        id: AllocId,
        alloc: Cow<'b, Allocation>,
        kind: Option<MemoryKind<Self::MemoryKind>>,
    ) -> (Cow<'b, Allocation<Self::PointerTag, Self::AllocExtra>>, Self::PointerTag) {
        // FIXME: fix
        T::init_allocation_extra(memory_extra, id, alloc, kind)
    }

    fn tag_global_base_pointer(memory_extra: &Self::MemoryExtra, id: AllocId) -> Self::PointerTag {
        // FIXME: fix
        T::tag_global_base_pointer(memory_extra, id)
    }

    fn init_frame_extra(
        ecx: &mut InterpCx<'mir, 'tcx, Self>,
        frame: Frame<'mir, 'tcx, Self::PointerTag>,
    ) -> InterpResult<'tcx, Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>> {
        ConstMachine::init_frame_extra(ecx, frame)
    }

    fn ptr_to_int(
        mem: &Memory<'mir, 'tcx, Self>,
        ptr: Pointer<Self::PointerTag>,
    ) -> InterpResult<'tcx, u64> {
        ConstMachine::ptr_to_int(mem, ptr)
    }

    fn access_local(
        ecx: &InterpCx<'mir, 'tcx, Self>,
        frame: &Frame<'mir, 'tcx, Self::PointerTag, Self::FrameExtra>,
        local: Local,
    ) -> InterpResult<'tcx, InterpOperand<Self::PointerTag>> {
        ConstMachine::access_local(ecx, frame, local)
    }
}
