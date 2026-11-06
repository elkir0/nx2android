namespace Ryujinx.Graphics.Shader
{
    /// <summary>
    /// nx2android: an NVN2 task or mesh shader, translated as a compute shader whose stage memory lies in shared memory (IGpuAccessor.QueryAmpereStageMemory), as the
    /// Vulkan task or mesh shader to write around it (CodeGen/Spirv/MeshStage.cs). The facts come from the shader's control section (dm/android/tools/shader_vk.py mesh_info,
    /// as dm/gfx/tools/dmsh_meshstage.py reads them for the Metal object and mesh functions).
    /// </summary>
    public sealed class MeshStageInfo
    {
        /// <summary>A task shader (true) or a mesh shader (false).</summary>
        public bool IsTask { get; init; }

        /// <summary>The bytes of the payload a task writes and its mesh shaders read.</summary>
        public int PayloadBytes { get; init; }

        /// <summary>A mesh shader launched by a task (its stage input is the task's payload); else by the draw itself.</summary>
        public bool HasTask { get; init; }

        /// <summary>A mesh shader's most vertices and primitives (triangles).</summary>
        public int MaxVertices { get; init; }
        public int MaxPrimitives { get; init; }

        /// <summary>A mesh shader's outputs in the order of its attribute blocks: (generic attribute, component), attribute -1 the position; the per-vertex ones, then the per-primitive ones.</summary>
        public (int Attribute, int Component)[] VertexOutputs { get; init; } = [];
        public (int Attribute, int Component)[] PrimitiveOutputs { get; init; } = [];

        /// <summary>Byte offsets in the stage memory: the workgroup's number, the stage input, the outputs, a mesh shader's primitive bytes (QueryAmpereStageMemory's).</summary>
        public int[] Layout { get; init; } = [];
    }
}
