use empa::buffer::{Buffer, Uniform, Usages};
use empa::device::Device;
use empa::type_flag::{O, X};

pub enum FallbackCounts<'a> {
    Binding(Uniform<'a, u32>, Uniform<'a, u32>),
    Buffer(
        Buffer<u32, Usages<O, O, O, X, O, O, O, O, O, O>>,
        Buffer<u32, Usages<O, O, O, X, O, O, O, O, O, O>>,
    ),
}

impl<'a> FallbackCounts<'a> {
    pub fn new(
        binding: Option<(Uniform<'a, u32>, Uniform<'a, u32>)>,
        device: &Device,
        fallback_counts: (u32, u32),
    ) -> Self {
        if let Some((b0, b1)) = binding {
            Self::Binding(b0, b1)
        } else {
            let b0 = device.create_buffer(fallback_counts.0, Usages::uniform_binding());
            let b1 = device.create_buffer(fallback_counts.1, Usages::uniform_binding());

            Self::Buffer(b0, b1)
        }
    }

    pub fn node_count(&self) -> Uniform<u32> {
        match self {
            FallbackCounts::Binding(b0, _) => b0.clone(),
            FallbackCounts::Buffer(b0, _) => b0.uniform(),
        }
    }

    pub fn edge_ref_count(&self) -> Uniform<u32> {
        match self {
            FallbackCounts::Binding(_, b1) => b1.clone(),
            FallbackCounts::Buffer(_, b1) => b1.uniform(),
        }
    }
}
