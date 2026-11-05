use candle_core::{Result, Tensor, D};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    pub temperature: f64,
    pub top_p: f64,
}

impl Sampler {
    pub fn new(temperature: f64, top_p: f64) -> Self {
        Self { temperature, top_p }
    }

    pub fn sample(&self, logits: &Tensor) -> Result<u32> {
        self.sample_with_rng(logits, &mut rand::rng())
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            candle_core::bail!("temperature must be finite and positive");
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            candle_core::bail!("top_p must be finite and in (0, 1]");
        }
        Ok(())
    }

    /// Sample using caller-owned RNG state; reuse it across generation steps.
    pub fn sample_with_rng<R: Rng + ?Sized>(&self, logits: &Tensor, rng: &mut R) -> Result<u32> {
        self.validate()?;
        let logits = logits.to_vec1::<f32>()?;
        if logits.is_empty()
            || logits.iter().any(|v| !v.is_finite())
            || logits.len() > u32::MAX as usize
        {
            candle_core::bail!("logits must be a nonempty finite vector with u32 token ids");
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut probs: Vec<(usize, f64)> = logits
            .iter()
            .enumerate()
            .map(|(i, v)| (i, ((*v as f64 - max) / self.temperature).exp()))
            .collect();
        probs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        let mut cumulative = 0.0f64;
        let mut cutoff = probs.len();
        for (idx, (_, p)) in probs.iter().enumerate() {
            cumulative += *p;
            if cumulative >= self.top_p * total {
                cutoff = idx + 1;
                break;
            }
        }
        let threshold = rng.random::<f64>() * cumulative;
        let mut running = 0.0;
        for &(token, weight) in &probs[..cutoff] {
            running += weight;
            if threshold < running {
                return Ok(token as u32);
            }
        }
        Ok(probs[cutoff - 1].0 as u32)
    }

    pub fn last_token_logits(logits: &Tensor) -> Result<Tensor> {
        let (_, seq, vocab) = logits.dims3()?;
        logits.narrow(D::Minus2, seq - 1, 1)?.reshape((vocab,))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    struct Fixed(u64);
    impl RngCore for Fixed {
        fn next_u32(&mut self) -> u32 {
            (self.0 >> 32) as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.0
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for chunk in dest.chunks_mut(8) {
                chunk.copy_from_slice(&self.0.to_le_bytes()[..chunk.len()]);
            }
        }
    }

    #[test]
    fn nucleus_draws_from_renormalized_retained_mass() -> Result<()> {
        let values = [0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()];
        let logits = Tensor::from_vec(values.to_vec(), 3, &Device::Cpu)?;
        let sampler = Sampler::new(1., 0.8);
        assert_eq!(
            sampler.sample_with_rng(&logits, &mut Fixed(u64::MAX / 4))?,
            0
        );
        assert_eq!(
            sampler.sample_with_rng(&logits, &mut Fixed(u64::MAX - 1))?,
            1
        );
        assert_eq!(
            Sampler::new(1., 0.5).sample_with_rng(&logits, &mut Fixed(u64::MAX - 1))?,
            0
        );
        assert_eq!(
            Sampler::new(1., 1.).sample_with_rng(&logits, &mut Fixed(u64::MAX - 1))?,
            2
        );
        assert_eq!(
            Sampler::new(0.1, 0.8).sample_with_rng(&logits, &mut Fixed(u64::MAX - 1))?,
            0
        );
        Ok(())
    }

    #[test]
    fn seeded_rng_state_is_reproducible_and_advances() -> Result<()> {
        let logits = Tensor::from_vec(vec![0f32; 4], 4, &Device::Cpu)?;
        let sampler = Sampler::new(1., 1.);
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        let left = (0..32)
            .map(|_| sampler.sample_with_rng(&logits, &mut a))
            .collect::<Result<Vec<_>>>()?;
        let right = (0..32)
            .map(|_| sampler.sample_with_rng(&logits, &mut b))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(left, right);
        assert!(left.iter().any(|v| *v != left[0]));
        Ok(())
    }

    #[test]
    fn equal_logits_tie_break_and_extreme_logits_are_stable() -> Result<()> {
        let tied = Tensor::from_vec(vec![0f32; 4], 4, &Device::Cpu)?;
        let sampler = Sampler::new(1., 0.5);
        assert_eq!(sampler.sample_with_rng(&tied, &mut Fixed(0))?, 0);
        assert_eq!(sampler.sample_with_rng(&tied, &mut Fixed(u64::MAX - 1))?, 1);
        let extremes = Tensor::from_vec(vec![f32::MAX, -f32::MAX], 2, &Device::Cpu)?;
        assert_eq!(
            Sampler::new(f64::MIN_POSITIVE, 1.)
                .sample_with_rng(&extremes, &mut Fixed(u64::MAX - 1))?,
            0
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_options_and_logits() -> Result<()> {
        let valid = Tensor::from_vec(vec![0f32], 1, &Device::Cpu)?;
        for (temp, top_p) in [
            (0., 1.),
            (-1., 1.),
            (f64::NAN, 1.),
            (f64::INFINITY, 1.),
            (1., 0.),
            (1., 1.1),
            (1., f64::NAN),
            (1., f64::INFINITY),
        ] {
            assert!(Sampler::new(temp, top_p)
                .sample_with_rng(&valid, &mut Fixed(0))
                .is_err());
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let bad = Tensor::from_vec(vec![value], 1, &Device::Cpu)?;
            assert!(Sampler::new(1., 1.)
                .sample_with_rng(&bad, &mut Fixed(0))
                .is_err());
        }
        let empty = Tensor::from_vec(Vec::<f32>::new(), 0, &Device::Cpu)?;
        assert!(Sampler::new(1., 1.)
            .sample_with_rng(&empty, &mut Fixed(0))
            .is_err());
        Ok(())
    }
}
