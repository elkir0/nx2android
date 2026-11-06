using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.StructuredIr;
using Ryujinx.Graphics.Shader.Translation;
using System;
using System.Collections.Generic;
using System.Linq;
using static Spv.Specification;

namespace Ryujinx.Graphics.Shader.CodeGen.Spirv
{
    using SpvInstruction = Spv.Generator.Instruction;
    using SpvLiteralInteger = Spv.Generator.LiteralInteger;

    // nx2android: an NVN2 task or mesh shader as a Vulkan task or mesh shader (VK_EXT_mesh_shader). The translator reads one as a compute shader whose stage memory lies in shared
    // memory (InstEmitAmpereMem.cs AIsberd, AStageOut; QueryAmpereStageMemory), and this writes that compute body as a function, called by an entry point of the TaskEXT or
    // MeshEXT model that fills the stage inputs and hands the stage outputs to Vulkan, as dm/gfx/tools/dmsh_meshstage.py wraps the same kernel as a Metal object or mesh function:
    //   the stage memory starts at zero; word 0 is the workgroup's number (a task's: the draw's first task + its workgroup; a mesh shader's with no task: the draw's first + its
    //   workgroup, the first from the uniform buffer mesh_first; with a task, its number among the task's launches, which also overwrites the input's word 0, as the hardware does);
    //   a mesh shader's input is its task's payload, copied in;
    //   after the body, a task's payload (the outputs, word 1 the mesh workgroups to launch) goes to the TaskPayloadWorkgroupEXT payload and EmitMeshTasksEXT launches them;
    //   a mesh shader's outputs are blocks of 32 vertices and 32 primitives, a word per vertex or primitive and component, the vertex components first, then the primitives'
    //   (MeshStageInfo), and its primitive bytes the count at +3, three indices a triangle from +4: all its vertices are written, its primitives up to the count.
    // A module with it is SPIR-V 1.4 (the extension's minimum): no BufferBlock, so its storage buffers are Block structures in the StorageBuffer class, and the entry point lists every global.
    static class MeshStage
    {
        private const ExecutionModel TaskEXT = (ExecutionModel)5364;
        private const ExecutionModel MeshEXT = (ExecutionModel)5365;
        private const ExecutionMode OutputPrimitivesEXT = (ExecutionMode)5270;
        private const ExecutionMode OutputTrianglesEXT = (ExecutionMode)5298;
        private const StorageClass TaskPayloadWorkgroupEXT = (StorageClass)5402;
        private const BuiltIn PrimitiveTriangleIndicesEXT = (BuiltIn)5296;
        private const Decoration PerPrimitiveEXT = (Decoration)5271;
        private const Capability MeshShadingEXT = (Capability)5283;

        private const int BlockSize = 32;   // vertices and primitives of an attribute block (dmsh_meshstage.py BLOCK)
        private const int MaxLaunches = 65535;   // mesh workgroups a task may launch: Vulkan's least maxMeshWorkGroupCount[0]

        public static bool IsOn(CodeGenContext context) => context.Properties.MeshStage != null;

        // Before the declarations
        public static void Begin(CodeGenContext context, StructuredProgramInfo info)
        {
            context.SetVersion(0x10400);
            context.StorageBufferClass = StorageClass.StorageBuffer;
            context.AddCapability(MeshShadingEXT);
            context.AddExtension("SPV_EXT_mesh_shader");

            info.IoDefinitions.Add(new IoDefinition(StorageKind.Input, IoVariable.ThreadId));
            info.IoDefinitions.Add(new IoDefinition(StorageKind.Input, IoVariable.CtaId));
        }

        // After the functions: the entry point around body, the translated program's main function
        public static void Generate(CodeGenContext context, SpvInstruction body)
        {
            MeshStageInfo m = context.Properties.MeshStage;
            int threads = context.Definitions.ComputeLocalSizeX * context.Definitions.ComputeLocalSizeY * context.Definitions.ComputeLocalSizeZ;
            int[] lay = m.Layout;

            if (lay.Length < 4 || lay.Any(o => (o & 3) != 0))
            {
                throw new InvalidOperationException($"Stage memory layout {string.Join(", ", lay)} is not one of words.");
            }

            int memId = -1;

            foreach ((int id, MemoryDefinition d) in context.Properties.SharedMemories)
            {
                if (d.Name == "shared_memory")
                {
                    memId = id;
                }
            }

            if (memId < 0)
            {
                throw new InvalidOperationException("A task or mesh shader without its stage memory.");
            }

            MemoryDefinition memDef = context.Properties.SharedMemories[memId];

            SpvInstruction u32 = context.TypeU32();
            SpvInstruction f32 = context.TypeFP32();
            SpvInstruction boolType = context.TypeBool();
            SpvInstruction v3u = context.TypeVector(u32, 3);
            SpvInstruction v4f = context.TypeVector(f32, 4);
            SpvInstruction voidType = context.TypeVoid();

            SpvInstruction U(int value) => context.Constant(u32, (uint)value);

            SpvInstruction mem = context.SharedMemories[memId];
            SpvInstruction Mem(SpvInstruction word) => context.AccessChain(context.TypePointer(StorageClass.Workgroup, u32), mem, word);

            int wgWord = lay[0] / 4, inWord = lay[1] / 4, outWord = lay[2] / 4, primByte = lay[3];
            int payWords = Math.Max(1, (m.PayloadBytes + 3) / 4);

            SpvInstruction payload = null;

            if (m.IsTask || m.HasTask)
            {
                payload = context.Variable(context.TypePointer(TaskPayloadWorkgroupEXT, context.TypeArray(u32, U(payWords))), TaskPayloadWorkgroupEXT);
                context.AddGlobalVariable(payload);
                context.Name(payload, "payload");
            }

            SpvInstruction Payload(SpvInstruction word) => context.AccessChain(context.TypePointer(TaskPayloadWorkgroupEXT, u32), payload, word);

            // A mesh shader's outputs: the position and each generic attribute a vertex, an attribute a primitive, three indices a triangle
            int maxV = Math.Max(1, m.MaxVertices), maxP = Math.Max(1, m.MaxPrimitives);
            var vertexOut = new SortedDictionary<int, SpvInstruction>();
            var primOut = new SortedDictionary<int, SpvInstruction>();
            SpvInstruction indices = null;

            SpvInstruction Output(int count, SpvInstruction elemType)
            {
                SpvInstruction v = context.Variable(context.TypePointer(StorageClass.Output, context.TypeArray(elemType, U(count))), StorageClass.Output);
                context.AddGlobalVariable(v);
                return v;
            }

            if (!m.IsTask)
            {
                foreach (int a in m.VertexOutputs.Select(o => o.Attribute).Distinct())
                {
                    SpvInstruction v = Output(maxV, v4f);
                    if (a < 0) context.Decorate(v, Decoration.BuiltIn, (SpvLiteralInteger)BuiltIn.Position);
                    else context.Decorate(v, Decoration.Location, (SpvLiteralInteger)a);
                    vertexOut[a] = v;
                }

                if (!vertexOut.ContainsKey(-1))
                {
                    throw new InvalidOperationException("A mesh shader whose SPH writes no position.");
                }

                foreach (int a in m.PrimitiveOutputs.Select(o => o.Attribute).Distinct())
                {
                    if (vertexOut.ContainsKey(a))
                    {
                        throw new InvalidOperationException($"Attribute {a} written per vertex and per primitive.");
                    }

                    SpvInstruction v = Output(maxP, v4f);
                    context.Decorate(v, Decoration.Location, (SpvLiteralInteger)a);
                    context.Decorate(v, PerPrimitiveEXT);
                    primOut[a] = v;
                }

                indices = Output(maxP, v3u);
                context.Decorate(indices, Decoration.BuiltIn, (SpvLiteralInteger)PrimitiveTriangleIndicesEXT);
            }

            SpvInstruction main = context.Function(voidType, FunctionControlMask.MaskNone, context.TypeFunction(voidType, false));
            context.Name(main, "main");
            context.AddFunction(main);
            context.AddLabel(context.Label());

            SpvInstruction counter = context.Variable(context.TypePointer(StorageClass.Function, u32), StorageClass.Function);
            context.AddLocalVariable(counter);

            SpvInstruction lid = context.CompositeExtract(u32, context.Load(v3u, context.Inputs[new IoDefinition(StorageKind.Input, IoVariable.ThreadId)]), (SpvLiteralInteger)0);
            SpvInstruction wg = context.CompositeExtract(u32, context.Load(v3u, context.Inputs[new IoDefinition(StorageKind.Input, IoVariable.CtaId)]), (SpvLiteralInteger)0);

            // for (i = lid; i < end; i += threads) work(i): every invocation takes its words
            void Loop(SpvInstruction end, Action<SpvInstruction> work)
            {
                SpvInstruction header = context.Label(), check = context.Label(), loopBody = context.Label(), next = context.Label(), merge = context.Label();
                context.Store(counter, lid);
                context.Branch(header);
                context.AddLabel(header);
                context.LoopMerge(merge, next, LoopControlMask.MaskNone);
                context.Branch(check);
                context.AddLabel(check);
                SpvInstruction i = context.Load(u32, counter);
                context.BranchConditional(context.ULessThan(boolType, i, end), loopBody, merge);
                context.AddLabel(loopBody);
                work(i);
                context.Branch(next);
                context.AddLabel(next);
                context.Store(counter, context.IAdd(u32, i, U(threads)));
                context.Branch(header);
                context.AddLabel(merge);
            }

            void If(SpvInstruction condition, Action work)
            {
                SpvInstruction then = context.Label(), merge = context.Label();
                context.SelectionMerge(merge, SelectionControlMask.MaskNone);
                context.BranchConditional(condition, then, merge);
                context.AddLabel(then);
                work();
                context.Branch(merge);
                context.AddLabel(merge);
            }

            void Barrier() => context.ControlBarrier(
                context.Constant(u32, Scope.Workgroup),
                context.Constant(u32, Scope.Workgroup),
                context.Constant(u32, MemorySemanticsMask.WorkgroupMemory | MemorySemanticsMask.AcquireRelease));

            SpvInstruction First() => context.Load(u32, context.AccessChain(context.TypePointer(StorageClass.Uniform, u32), context.ConstantBuffers[context.Properties.MeshFirstBinding],
                context.Constant(context.TypeS32(), 0), context.Constant(context.TypeS32(), 0)));

            SpvInstruction isFirst = context.IEqual(boolType, lid, U(0));

            Loop(U(memDef.ArrayLength), i => context.Store(Mem(i), U(0)));
            Barrier();

            if (m.IsTask || !m.HasTask)
            {
                If(isFirst, () => context.Store(Mem(U(wgWord)), context.IAdd(u32, First(), wg)));
            }
            else
            {
                Loop(U(payWords), i => context.Store(Mem(context.IAdd(u32, U(inWord), i)), context.Load(u32, Payload(i))));
                Barrier();
                If(isFirst, () =>
                {
                    context.Store(Mem(U(wgWord)), wg);
                    context.Store(Mem(U(inWord)), wg);
                });
            }

            Barrier();
            context.FunctionCall(voidType, body);
            Barrier();

            if (m.IsTask)
            {
                Loop(U(payWords), i => context.Store(Payload(i), context.Load(u32, Mem(context.IAdd(u32, U(outWord), i)))));
                Barrier();
                SpvInstruction launches = context.GlslUMin(u32, context.Load(u32, Mem(U(outWord + 1))), U(MaxLaunches));
                context.EmitMeshTasksEXT(launches, U(1), U(1), payload);
            }
            else
            {
                int stride = Math.Max(m.VertexOutputs.Length + m.PrimitiveOutputs.Length, 1) * BlockSize;
                SpvInstruction primitives = context.GlslUMin(u32, context.ShiftRightLogical(u32, context.Load(u32, Mem(U(primByte / 4))), U(24)), U(maxP));
                context.SetMeshOutputsEXT(U(maxV), primitives);

                // the first word of vertex or primitive n's block slot
                SpvInstruction Slot(SpvInstruction n) => context.IAdd(u32, U(outWord),
                    context.IAdd(u32, context.IMul(u32, context.UDiv(u32, n, U(BlockSize)), U(stride)), context.UMod(u32, n, U(BlockSize))));

                // a vec4 of an output's components, (0, 0, 0, 1) where the shader writes none, as a vertex output starts
                void Write(SpvInstruction target, SpvInstruction n, SpvInstruction slot, (int Attribute, int Component)[] list, int at, int attribute)
                {
                    SpvInstruction[] c = new SpvInstruction[4];

                    for (int k = 0; k < 4; k++)
                    {
                        int j = Array.IndexOf(list, (attribute, k));
                        c[k] = j < 0 ? context.Constant(f32, k == 3 ? 1f : 0f) : context.Bitcast(f32, context.Load(u32, Mem(context.IAdd(u32, slot, U((at + j) * BlockSize)))));
                    }

                    context.Store(context.AccessChain(context.TypePointer(StorageClass.Output, v4f), target, n), context.CompositeConstruct(v4f, c));
                }

                Loop(U(maxV), n =>
                {
                    SpvInstruction slot = Slot(n);
                    foreach ((int a, SpvInstruction v) in vertexOut) Write(v, n, slot, m.VertexOutputs, 0, a);
                });

                if (primOut.Count != 0)
                {
                    Loop(primitives, n =>
                    {
                        SpvInstruction slot = Slot(n);
                        foreach ((int a, SpvInstruction v) in primOut) Write(v, n, slot, m.PrimitiveOutputs, m.VertexOutputs.Length, a);
                    });
                }

                // triangle n: its three index bytes, stored whole as the triangle's uvec3 (an index past the vertices is the last vertex: Vulkan leaves one past the count undefined).
                // Not a byte at a time into component n % 3: a component chosen at run time of a PrimitiveTriangleIndicesEXT element is a form glslang never writes, and the first
                // mesh pipeline of Mario Kart World (mesh shader 30df8b09fea2f84b) crashed inside the Adreno 840's compiler (libllvm-qgl.so, a read at 0x10) with it
                Loop(primitives, n =>
                {
                    SpvInstruction first = context.IAdd(u32, U(primByte + 4), context.IMul(u32, n, U(3)));
                    SpvInstruction[] corner = new SpvInstruction[3];

                    for (int k = 0; k < 3; k++)
                    {
                        SpvInstruction at = context.IAdd(u32, first, U(k));
                        SpvInstruction word = context.Load(u32, Mem(context.ShiftRightLogical(u32, at, U(2))));
                        SpvInstruction index = context.BitwiseAnd(u32, context.ShiftRightLogical(u32, word, context.ShiftLeftLogical(u32, context.BitwiseAnd(u32, at, U(3)), U(3))), U(0xff));
                        corner[k] = context.GlslUMin(u32, index, U(maxV - 1));
                    }

                    context.Store(context.AccessChain(context.TypePointer(StorageClass.Output, v3u), indices, n), context.CompositeConstruct(v3u, corner));
                });

                context.Return();
            }

            context.FunctionEnd();

            context.AddEntryPoint(m.IsTask ? TaskEXT : MeshEXT, main, "main", context.GlobalVariables.ToArray());
            context.AddExecutionMode(main, ExecutionMode.LocalSize, (SpvLiteralInteger)context.Definitions.ComputeLocalSizeX,
                (SpvLiteralInteger)context.Definitions.ComputeLocalSizeY, (SpvLiteralInteger)context.Definitions.ComputeLocalSizeZ);

            if (!m.IsTask)
            {
                context.AddExecutionMode(main, ExecutionMode.OutputVertices, (SpvLiteralInteger)maxV);
                context.AddExecutionMode(main, OutputPrimitivesEXT, (SpvLiteralInteger)maxP);
                context.AddExecutionMode(main, OutputTrianglesEXT);
            }
        }
    }
}
