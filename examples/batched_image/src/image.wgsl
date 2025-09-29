struct Globals {
    transform: mat4x4<f32>;
    scale_factor: f32;
    _padding: vec3<f32>;
}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position_and_size: vec4<f32>,
    @location(1) atlas_uvs: vec4<f32>,
    @location(2) layer_opacity: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) quad_coord: vec2<f32>,
    @location(1) uv_min: vec2<f32>,
    @location(2) uv_max: vec2<f32>,
    @location(3) atlas_layer: f32,
    @location(4) opacity: f32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var atlas_sampler: sampler;
@group(1) @binding(0) var atlas_texture: texture_2d_array<f32>;

fn quad_position(vertex: u32) -> vec2<f32> {
    let x = f32(vertex & 1u);
    let y = f32((vertex >> 1u) & 1u);
    vec2<f32>(x, y)
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let local = quad_position(input.vertex_index);
    let size = input.position_and_size.zw;
    let origin = input.position_and_size.xy;

    let logical = origin + local * size;
    let clip = globals.transform * vec4<f32>(logical, 0.0, 1.0);

    out.clip_position = clip;
    out.quad_coord = local;
    out.uv_min = input.atlas_uvs.xy;
    out.uv_max = input.atlas_uvs.zw;
    out.atlas_layer = input.layer_opacity.x;
    out.opacity = input.layer_opacity.y;

    out
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = mix(input.uv_min, input.uv_max, input.quad_coord);
    let color = textureSample(atlas_texture, atlas_sampler, vec3<f32>(uv, input.atlas_layer));
    vec4<f32>(color.rgb, color.a * input.opacity)
}
