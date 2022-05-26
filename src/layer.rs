use rand::Rng;
use nannou::rand::random_range;

pub struct Layer {
    pub n_inputs: usize,
    pub n_neurons: usize,

    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,

    pub outputs: Vec<f64>,
}

impl Layer {
    fn dot(v1: Vec::<f64>, v2: Vec::<f64>) -> f64 {
        // take the dot product of two vectors
        let mut sum = 0.0;
        for i in 0..v1.len() {
            sum += v1[i] * v2[i];
        }
        sum
    }

    pub fn new(n_inputs: usize, n_neurons: usize) -> Layer {
        Layer {

            n_inputs: {
                n_inputs
            },

            n_neurons: {
                n_neurons
            },

            weights: {
                let mut rng = rand::thread_rng();
                let mut v = Vec::<Vec<f64>>::new();
                for _ in 0..n_neurons {
                    v.push(Vec::<f64>::new());
                    for _ in 0..n_inputs {
                        v
                            .last_mut()
                            .unwrap()
                            .push(
                                rng.gen_range(-5.0..5.0) * random_range(0.0, 1.0)
                            )
                    }
                }
                v
            },

            biases: {
                let mut b = Vec::<f64>::new();
                let mut rng = rand::thread_rng();
                for _ in 0..n_neurons {
                    b.push(
                        rng.gen_range(-5.0..5.0) * random_range(0.0, 1.0)
                    )
                }
                b
            },

            outputs: {
                Vec::<f64>::new()
            }

        }
    }

    pub fn new_with_values(
        n_inputs: usize,
        n_neurons: usize,
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
    ) -> Layer {
        Layer {
            n_inputs: {
                n_inputs
            },
            n_neurons: {
                n_neurons
            },
            weights: {
                weights
            },
            biases: {
                biases
            },
            outputs: {
                Vec::<f64>::new()
            }
        }
    }

    pub fn copy(&self) -> Layer {
        let n_inputs = self.n_inputs;
        let n_neurons = self.n_neurons;
        let weights = self.weights.clone();
        let biases = self.biases.clone();
        let outputs = self.outputs.clone();

        Layer {
            n_inputs,
            n_neurons,
            weights,
            biases,
            outputs,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) {
        self.outputs.clear();
        let mut i = 0;
        for weight_group in &self.weights {
            self.outputs.push(
                Layer::dot(
                    weight_group.to_vec(),
                    inputs.to_vec()
                ) + self.biases[i]
            );
            i += 1;
        }
    }

    pub fn relu(&mut self) {
        self.outputs.iter_mut().for_each(|x| {
            if *x < 0.0 {
                *x = 0.0;
            }
        });
    }

    pub fn softmax(&mut self) {
        // make safe inputs
        let mut max = self.outputs[0];
        self.outputs.iter().for_each(|x| {
            if *x > max {
                max = *x;
            }
        });
        self.outputs.iter_mut().for_each(|x| {
            *x -= max;
        });

        // exponentiate
        self.outputs.iter_mut().for_each(|x| {
            *x = (*x).exp();
        });

        // normalized base
        let normalize_base = self.outputs.iter().sum::<f64>();

        // normalize all values by dividing by the normalized base
        self.outputs.iter_mut().for_each(|x| {
            *x /= normalize_base;
        });
    }
}