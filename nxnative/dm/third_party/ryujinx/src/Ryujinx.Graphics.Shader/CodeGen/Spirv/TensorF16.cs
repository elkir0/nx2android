using Ryujinx.Graphics.Shader.StructuredIr;
using Ryujinx.Graphics.Shader.Translation;
using static Spv.Specification;

namespace Ryujinx.Graphics.Shader.CodeGen.Spirv
{
    using SpvInstruction = Spv.Generator.Instruction;
    using SpvLiteralInteger = Spv.Generator.LiteralInteger;

    // nx2android: the body of dmsh_hmma16816_f16 (Ampere HMMA.16816 with an f16 accumulator, HelperFunctionManager.GenerateHmma16816F16) in packed half arithmetic, where the host asks
    // for it (ShaderProperties.HmmaF16, IGpuAccessor.QueryHostHmmaF16). The translator's body converts every half to float and makes 64 FP32 FMAs a lane an HMMA, in k order: on the
    // Adreno 840 that is 128 of its ~190 instructions, and agl's NDSR upscaler runs 144 to 288 HMMAs a thread. Here each lane takes its A row and B column words from the same
    // scratch block as packed half pairs and makes 8 two-wide half FMAs an output (the even and odd k apart), then adds the two halves and C in float: 32 FMAs and no conversion of
    // A and B. A sum in half precision, as the console's f16 accumulator allows (PTX mma: at least the accumulator's precision), not the translator's float sum: not bit-exact, so
    // a switch. C is not stored to the block (only the Metal rewrite of the body reads it there)
    static class TensorF16
    {
        public static bool TryGenerate(CodeGenContext context, StructuredFunction function)
        {
            if (!context.Properties.HmmaF16 || function.Name != "dmsh_hmma16816_f16" || SharedMemorySpill.IsOn(context) ||
                function.InArguments.Length != 10 || function.OutArguments.Length != 1)
            {
                return false;
            }

            SpvInstruction scratch = null;

            foreach ((int id, MemoryDefinition memory) in context.Properties.SharedMemories)
            {
                if (memory.Name == "hmma_scratch" && context.SharedMemories.TryGetValue(id, out SpvInstruction v))
                {
                    scratch = v;
                }
            }

            if (scratch == null)
            {
                return false;
            }

            context.AddCapability(Capability.Float16);

            SpvInstruction u32 = context.TypeU32();
            SpvInstruction s32 = context.TypeS32();
            SpvInstruction f32 = context.TypeFP32();
            SpvInstruction h16 = context.TypeFloat(16);
            SpvInstruction h2 = context.TypeVector(h16, 2);
            SpvInstruction f2 = context.TypeVector(f32, 2);
            SpvInstruction pw = context.TypePointer(StorageClass.Workgroup, u32);

            SpvInstruction U(uint v) => context.Constant(u32, v);
            SpvInstruction Arg(int i) => context.Bitcast(u32, context.Load(s32, context.GetArgumentPointer(new AstOperand(IntermediateRepresentation.OperandType.Argument, i))));
            SpvInstruction Word(SpvInstruction index) => context.AccessChain(pw, scratch, index);

            SpvInstruction lane = Arg(0), bse = Arg(1);
            SpvInstruction g = context.ShiftRightLogical(u32, lane, U(2)), t = context.BitwiseAnd(u32, lane, U(3)), g8 = context.IAdd(u32, g, U(8));

            SpvInstruction At(SpvInstruction row, uint word) => context.IAdd(u32, bse, context.IAdd(u32, context.ShiftLeftLogical(u32, row, U(3)), context.IAdd(u32, t, U(word))));

            // A row-major (words 0..127, 8 a row), B transposed (128..191), as the translator's body lays them out
            context.Store(Word(At(g, 0)), Arg(2));
            context.Store(Word(At(g8, 0)), Arg(3));
            context.Store(Word(At(g, 4)), Arg(4));
            context.Store(Word(At(g8, 4)), Arg(5));
            context.Store(Word(context.IAdd(u32, U(128), At(g, 0))), Arg(6));
            context.Store(Word(context.IAdd(u32, U(128), At(g, 4))), Arg(7));

            SpvInstruction subgroup = U((uint)Scope.Subgroup);
            SpvInstruction semantics = U((uint)(MemorySemanticsMask.WorkgroupMemory | MemorySemanticsMask.AcquireRelease));

            context.ControlBarrier(subgroup, subgroup, semantics);

            SpvInstruction[] d = new SpvInstruction[2];

            for (int q = 0; q < 2; q++)
            {
                SpvInstruction rowBase = context.IAdd(u32, bse, context.ShiftLeftLogical(u32, q == 0 ? g : g8, U(3)));
                SpvInstruction c = context.GlslUnpackHalf2x16(f2, Arg(8 + q));
                SpvInstruction[] o = new SpvInstruction[2];

                for (int j = 0; j < 2; j++)
                {
                    SpvInstruction col = context.IAdd(u32, context.ShiftLeftLogical(u32, t, U(1)), U((uint)j));
                    SpvInstruction colBase = context.IAdd(u32, bse, context.IAdd(u32, U(128), context.ShiftLeftLogical(u32, col, U(3))));
                    SpvInstruction acc = context.ConstantNull(h2);

                    for (uint kw = 0; kw < 8; kw++)
                    {
                        SpvInstruction aw = context.Bitcast(h2, context.Load(u32, Word(context.IAdd(u32, rowBase, U(kw)))));
                        SpvInstruction bw = context.Bitcast(h2, context.Load(u32, Word(context.IAdd(u32, colBase, U(kw)))));
                        acc = context.GlslFma(h2, aw, bw, acc);
                    }

                    SpvInstruction even = context.FConvert(f32, context.CompositeExtract(h16, acc, (SpvLiteralInteger)0));
                    SpvInstruction odd = context.FConvert(f32, context.CompositeExtract(h16, acc, (SpvLiteralInteger)1));
                    o[j] = context.FAdd(f32, context.CompositeExtract(f32, c, (SpvLiteralInteger)j), context.FAdd(f32, even, odd));
                }

                d[q] = context.GlslPackHalf2x16(u32, context.CompositeConstruct(f2, o[0], o[1]));
            }

            // The next HMMA of this subgroup overwrites the block: every lane has read it by then
            context.ControlBarrier(subgroup, subgroup, semantics);

            context.Store(context.GetArgumentPointer(new AstOperand(IntermediateRepresentation.OperandType.Argument, 10)), context.Bitcast(s32, d[1]));
            context.ReturnValue(context.Bitcast(s32, d[0]));

            return true;
        }
    }
}
