using Ryujinx.Graphics.Shader.StructuredIr;
using Ryujinx.Graphics.Shader.Translation;

using static Ryujinx.Graphics.Shader.CodeGen.Glsl.Instructions.InstGenHelper;
using static Ryujinx.Graphics.Shader.StructuredIr.InstructionInfo;

namespace Ryujinx.Graphics.Shader.CodeGen.Glsl.Instructions
{
    static class InstGenBallot
    {
        public static string Ballot(CodeGenContext context, AstOperation operation)
        {
            AggregateType dstType = GetSrcVarType(operation.Inst, 0);

            string arg = GetSourceExpr(context, operation.GetSource(0), dstType);

            // nx2android: a second source picks the word at run time (EmitterContext.BallotWord)
            string component = operation.SourcesCount > 1
                ? $"[{GetSourceExpr(context, operation.GetSource(1), GetSrcVarType(operation.Inst, 1))}]"
                : $".{"xyzw"[operation.Index]}";

            if (context.HostCapabilities.SupportsShaderBallot)
            {
                return $"unpackUint2x32(ballotARB({arg})){component}";
            }
            else
            {
                return $"subgroupBallot({arg}){component}";
            }
        }
    }
}
