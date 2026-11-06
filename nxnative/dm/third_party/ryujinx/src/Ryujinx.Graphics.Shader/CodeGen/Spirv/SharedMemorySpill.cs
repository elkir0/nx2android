using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.StructuredIr;
using Ryujinx.Graphics.Shader.Translation;
using System;
using System.Collections.Generic;
using static Spv.Specification;

namespace Ryujinx.Graphics.Shader.CodeGen.Spirv
{
    using SpvInstruction = Spv.Generator.Instruction;
    using SpvLiteralInteger = Spv.Generator.LiteralInteger;

    // nx2android: a compute shader's shared memory in a storage buffer, when it is more than the host gives a workgroup (ShaderProperties.SharedMemorySpillBinding;
    // IGpuAccessor.QueryHostSharedMemoryLimit). Ampere gives a workgroup up to 100 KB, Adreno 32 KB, and Vulkan refuses a pipeline that declares more: Mario Kart World's
    // DrivePassShaderDefault declares 48 KB in 168 compute shaders. The shared memories lie one after the other in a slice of the buffer, a slice a workgroup: workgroup
    // ((z * count y) + y) * count x + x of the dispatch takes slice (that number modulo the slices), so a dispatch of any size runs as long as workgroups that far apart do not
    // run together (an indirect dispatch's count is not known on the CPU). The barriers then order buffer memory too. As dm/gfx/tools/dmsh_build.py tg_spill does for Metal.
    static class SharedMemorySpill
    {
        public static bool IsOn(CodeGenContext context) => context.Properties.SharedMemorySpillBinding >= 0;

        // Instead of DeclareMemories for the workgroup storage class: each shared memory's first word in the slice, the variable for the slice's first word, NumWorkgroups
        public static void Declare(CodeGenContext context)
        {
            // The workgroup's invocations write the slice and read each other's words across the barriers. Under the GLSL450 memory model a buffer that is not Coherent
            // gives no such visibility, barriers or not: a device may serve its reads from a per-core cache the other invocations' writes never reach. Workgroup memory
            // is coherent by nature, so the buffer standing in for it is declared so (DeclareStorageBuffers has made its variable by now)
            context.Decorate(context.StorageBuffers[context.Properties.SharedMemorySpillBinding], Decoration.Coherent);

            int offset = 0;

            foreach ((int id, MemoryDefinition memory) in context.Properties.SharedMemories)
            {
                if (memory.Type != (AggregateType.Array | AggregateType.U32))
                {
                    throw new InvalidOperationException($"Shared memory \"{memory.Name}\" of type {memory.Type} cannot move to a buffer.");
                }

                context.SharedMemorySpillOffsets[id] = offset;
                offset += memory.ArrayLength;
            }

            // Its first words in workgroup memory (ShaderProperties.SharedMemorySpillKeep: one shared memory, at offset 0 of the slice; the slice holds the words past them)
            if (context.Properties.SharedMemorySpillKeep > 0)
            {
                SpvInstruction low = context.TypeArray(context.TypeU32(), context.Constant(context.TypeU32(), (uint)context.Properties.SharedMemorySpillKeep));

                context.SharedMemorySpillLow = context.Variable(context.TypePointer(StorageClass.Workgroup, low), StorageClass.Workgroup);
                context.AddGlobalVariable(context.SharedMemorySpillLow);
                context.Name(context.SharedMemorySpillLow, "shared_memory");
            }

            context.SharedMemorySpillBase = context.Variable(context.TypePointer(StorageClass.Private, context.TypeU32()), StorageClass.Private);
            context.AddGlobalVariable(context.SharedMemorySpillBase);
            context.Name(context.SharedMemorySpillBase, "shared_spill_base");

            context.SharedMemorySpillCounts = context.Variable(context.TypePointer(StorageClass.Input, context.TypeVector(context.TypeU32(), 3)), StorageClass.Input);
            context.AddGlobalVariable(context.SharedMemorySpillCounts);
            context.Decorate(context.SharedMemorySpillCounts, Decoration.BuiltIn, (SpvLiteralInteger)BuiltIn.NumWorkgroups);
            context.ExtraInterface.Add(context.SharedMemorySpillCounts);
        }

        // At the entry point's start, before any shared memory access: the slice's first word
        public static void EmitBase(CodeGenContext context)
        {
            if (!IsOn(context))
            {
                return;
            }

            SpvInstruction u32 = context.TypeU32();
            SpvInstruction v3 = context.TypeVector(u32, 3);
            SpvInstruction id = context.Load(v3, context.Inputs[new IoDefinition(StorageKind.Input, IoVariable.CtaId)]);
            SpvInstruction count = context.Load(v3, context.SharedMemorySpillCounts);

            SpvInstruction X(SpvInstruction v, int c) => context.CompositeExtract(u32, v, (SpvLiteralInteger)c);

            SpvInstruction linear = context.IAdd(u32, context.IMul(u32, context.IAdd(u32, context.IMul(u32, X(id, 2), X(count, 1)), X(id, 1)), X(count, 0)), X(id, 0));
            SpvInstruction slice = context.UMod(u32, linear, context.Constant(u32, (uint)context.Properties.SharedMemorySpillSlices));

            context.Store(context.SharedMemorySpillBase, context.IMul(u32, slice, context.Constant(u32, (uint)context.Properties.SharedMemorySpillWords)));
        }

        // A pointer to word index of shared memory id
        public static SpvInstruction Pointer(CodeGenContext context, int id, SpvInstruction index)
        {
            SpvInstruction u32 = context.TypeU32();
            SpvInstruction word = context.IAdd(u32, context.Load(u32, context.SharedMemorySpillBase),
                context.IAdd(u32, context.Constant(u32, (uint)context.SharedMemorySpillOffsets[id]), context.Bitcast(u32, index)));

            return context.AccessChain(context.TypePointer(context.StorageBufferClass, u32), context.StorageBuffers[context.Properties.SharedMemorySpillBinding],
                context.Constant(context.TypeS32(), 0), word);
        }

        // A barrier's memory semantics: buffer memory too, where the shared memory lies
        public static MemorySemanticsMask Semantics(CodeGenContext context, MemorySemanticsMask semantics)
        {
            return IsOn(context) ? semantics | MemorySemanticsMask.UniformMemory : semantics;
        }

        // A shared memory access whose word lies in workgroup memory or in the buffer, chosen at run time (ShaderProperties.SharedMemorySpillKeep)
        public static bool IsSplit(CodeGenContext context, AstOperation operation)
        {
            return operation.StorageKind == StorageKind.SharedMemory && IsOn(context) && context.Properties.SharedMemorySpillKeep > 0;
        }

        // Its word index (the operand after the shared memory's id), evaluated before the access's other operands, as GetStoragePointer does
        public static SpvInstruction SplitIndex(CodeGenContext context, AstOperation operation)
        {
            int values = operation.Inst == Instruction.Store || operation.Inst.IsAtomic() ? 1 : 0;

            if (operation.SourcesCount - 1 - values - (operation.Inst == Instruction.AtomicCompareAndSwap ? 1 : 0) != 1)
            {
                throw new InvalidOperationException($"{operation.Inst} on shared memory in a buffer with other than one index.");
            }

            return context.Bitcast(context.TypeU32(), context.Get(AggregateType.S32, operation.GetSource(1)));
        }

        // The access emit(pointer) on the word index: in workgroup memory below the words kept there, else in the slice at index - kept. Both arms make it, a phi takes
        // its result (resultType null: a store, none)
        public static SpvInstruction Split(CodeGenContext context, SpvInstruction index, SpvInstruction resultType, Func<SpvInstruction, SpvInstruction> emit)
        {
            SpvInstruction u32 = context.TypeU32();
            SpvInstruction keep = context.Constant(u32, (uint)context.Properties.SharedMemorySpillKeep);
            SpvInstruction low = context.Label();
            SpvInstruction high = context.Label();
            SpvInstruction merge = context.Label();
            SpvInstruction inLow = context.ULessThan(context.TypeBool(), index, keep);

            context.SelectionMerge(merge, SelectionControlMask.MaskNone);
            context.BranchConditional(inLow, low, high);

            context.AddLabel(low);
            SpvInstruction lowResult = emit(context.AccessChain(context.TypePointer(StorageClass.Workgroup, u32), context.SharedMemorySpillLow, index));
            context.Branch(merge);

            context.AddLabel(high);
            SpvInstruction word = context.IAdd(u32, context.Load(u32, context.SharedMemorySpillBase), context.ISub(u32, index, keep));
            SpvInstruction highResult = emit(context.AccessChain(context.TypePointer(context.StorageBufferClass, u32), context.StorageBuffers[context.Properties.SharedMemorySpillBinding],
                context.Constant(context.TypeS32(), 0), word));
            context.Branch(merge);

            context.AddLabel(merge);

            return resultType != null ? context.Phi(resultType, lowResult, low, highResult, high) : null;
        }
    }

    partial class CodeGenContext
    {
        // nx2android: SharedMemorySpill's state
        public Dictionary<int, int> SharedMemorySpillOffsets { get; } = new();
        public SpvInstruction SharedMemorySpillBase { get; set; }
        public SpvInstruction SharedMemorySpillCounts { get; set; }
        public SpvInstruction SharedMemorySpillLow { get; set; }

        // nx2android: entry point interface variables that are no input or output of the shader's own (SharedMemorySpill's NumWorkgroups)
        public List<SpvInstruction> ExtraInterface { get; } = [];

        // nx2android: the storage class of storage buffers: Uniform with BufferBlock in SPIR-V 1.3, StorageBuffer with Block in a SPIR-V 1.4 module (MeshStage)
        public StorageClass StorageBufferClass { get; set; } = StorageClass.Uniform;
    }
}
