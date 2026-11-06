using System.Collections.Generic;
using static Spv.Specification;

namespace Spv.Generator
{
    // nx2android: what SPV_EXT_mesh_shader needs and the 2020 grammar this generator was made from does not have (Ryujinx.Graphics.Shader CodeGen/Spirv/MeshStage.cs):
    // its two instructions, and a module of SPIR-V 1.4 (the extension's minimum), whose entry point lists every global variable it uses
    public partial class Module
    {
        public const Op OpEmitMeshTasksEXT = (Op)5294;
        public const Op OpSetMeshOutputsEXT = (Op)5295;

        public void SetVersion(uint version)
        {
            _version = version;
        }

        public bool HasCapability(Capability capability)
        {
            return _capabilities.Contains(capability);
        }

        public IReadOnlyList<Instruction> GlobalVariables => _globals;

        // A terminator: the task workgroup's last instruction, which every invocation executes in uniform control flow
        public Instruction EmitMeshTasksEXT(Instruction groupCountX, Instruction groupCountY, Instruction groupCountZ, Instruction payload)
        {
            Instruction result = NewInstruction(OpEmitMeshTasksEXT);

            result.AddOperand(groupCountX);
            result.AddOperand(groupCountY);
            result.AddOperand(groupCountZ);

            if (payload != null)
            {
                result.AddOperand(payload);
            }

            AddToFunctionDefinitions(result);

            return result;
        }

        public Instruction SetMeshOutputsEXT(Instruction vertexCount, Instruction primitiveCount)
        {
            Instruction result = NewInstruction(OpSetMeshOutputsEXT);

            result.AddOperand(vertexCount);
            result.AddOperand(primitiveCount);
            AddToFunctionDefinitions(result);

            return result;
        }
    }
}
