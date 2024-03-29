#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "raycommon.glsl"
#include "wavefront.glsl"
#include "sampling.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices {ivec3 i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MatIndices {int i[]; }; // Material ID for each triangle
layout(set = 1, binding = eObjDescs, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;
layout(set = 1, binding = eTextures) uniform sampler2D textureSamplers[];
layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };

layout(location = 1) rayPayloadInEXT vec3 hitValue;

hitAttributeEXT vec3 attribs;

void main()
{
    // Object data
    ObjDesc    objResource = objDesc.i[gl_InstanceCustomIndexEXT];
    MatIndices matIndices  = MatIndices(objResource.materialIndexAddress);
    Materials  materials   = Materials(objResource.materialAddress);
    Indices    indices     = Indices(objResource.indexAddress);
    Vertices   vertices    = Vertices(objResource.vertexAddress);
  
    // Indices of the triangle
    ivec3 ind = indices.i[gl_PrimitiveID];
  
    // Vertex of the triangle
    Vertex v0 = vertices.v[ind.x];
    Vertex v1 = vertices.v[ind.y];
    Vertex v2 = vertices.v[ind.z];

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Computing the normal at hit position
    const vec3 nrm      = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
    const vec3 worldNrm = normalize(vec3(nrm * gl_WorldToObjectEXT));  // Transforming the normal to world space

    prd.hitNorm = worldNrm;

    // https://en.wikipedia.org/wiki/Path_tracing
    // Material of the object
    int               matIdx = matIndices.i[gl_PrimitiveID];
    WaveFrontMaterial mat    = materials.m[matIdx];

    // Pick a random direction from here and keep going.
    vec3 tangent, bitangent;
    createCoordinateSystem(worldNrm, tangent, bitangent);
    vec3 rayOrigin    = worldPos;
    vec3 rayDirection = samplingHemisphere(prd.seed, tangent, bitangent, worldNrm);

    // Probability of the newRay (cosine distributed)
    const float p = 1 / M_PI;

    // Compute the BRDF for this ray (assuming Lambertian reflection)
    float cos_theta = dot(rayDirection, worldNrm);
    vec3  BRDF      = mat.diffuse / M_PI;
    
    prd.rayOrigin    = rayOrigin;
    prd.rayDirection = rayDirection;
    prd.hitValue     = mat.emission;
    prd.weight       = BRDF * cos_theta / p;
    return;

}
