use crate::layer::Layer;

use nannou::prelude::*;
use nannou::color::Alpha;
use rand::seq::SliceRandom;

pub struct Rocket {
    pub pos: Vec2,
    vel: Vec2,
    acc: Vec2,

    pub target: Vec2,

    pub layer1: Layer,
    pub layer2: Layer,
    pub layer3: Layer,
    pub layer4: Layer,

    color: Alpha<Rgb<f64>, f64>,
}

impl Rocket {
    pub fn new(_width: u32, height: u32, target: Vec2) -> Rocket {
        Rocket {

            pos: {
                vec2(0.0, -(height as f32) / 2.0)
            },

            vel: {
                vec2(0.0, 0.0)
            },

            acc: {
                vec2(0.0, 0.0)
            },

            layer1: {
                Layer::new(4, 3)
            },

            layer2: {
                Layer::new(3, 3)
            },

            layer3: {
                Layer::new(3, 3)
            },

            layer4: {
                Layer::new(3, 3)
            },

            target: {
                target
            },

            color: {
                nannou::color::rgba(1.0, 1.0, 1.0, 1.0)
            },

        }
    }

    pub fn new_with_values(
        pos: Vec2,
        vel: Vec2,
        acc: Vec2,
        layer1: Layer,
        layer2: Layer,
        layer3: Layer,
        layer4: Layer,
        target: Vec2,
        color: Alpha<Rgb<f64>, f64>
    ) -> Rocket {
        Rocket {
            pos,
            vel,
            acc,
            layer1,
            layer2,
            layer3,
            layer4,
            target,
            color,
        }
    }

    pub fn copy(&self) -> Rocket {
        let pos = self.pos;
        let vel = self.vel;
        let acc = self.acc;
        let layer1 = self.layer1.copy();
        let layer2 = self.layer2.copy();
        let layer3 = self.layer3.copy();
        let layer4 = self.layer4.copy();
        let target = self.target;
        let color = self.color;
        Rocket {
            pos,
            vel,
            acc,
            layer1,
            layer2,
            layer3,
            layer4,
            target,
            color,
        }
    }

    pub fn evaluate(&self, width: u32) -> u32 {
        let d = self.pos.distance(self.target);
        map_range(d, 0.0, width as f32, 100.0, 0.0) as u32
    }

    pub fn cross_over_layer(layer1: Layer, layer2: Layer) -> (Vec::<Vec<f64>>, Vec::<f64>) {
        let mut new_weights = Vec::<Vec<f64>>::new();
        let mut new_biases = Vec::<f64>::new();
        let mut midpoint = random_range(0, layer1.n_neurons);

        for i in 0..(layer1.n_neurons) {
            new_weights.push(
                {
                    if i > midpoint {
                        layer1.weights[i].clone()
                    } else {
                        layer2.weights[i].clone()
                    }
                }
            )
        }

        midpoint = random_range(0, layer1.n_neurons);
    
        for i in 0..(layer1.n_neurons) {
            new_biases.push(
                {
                    if i > midpoint {
                        layer1.biases[i]
                    } else {
                        layer2.biases[i]
                    }
                }
            )
        }

        (new_weights, new_biases)
    }

    pub fn cross_over(&self, other: &Rocket, height: u32) -> Rocket {
        let (l1_weights, l1_biases) = Rocket::cross_over_layer(
            self.layer1.copy(),
            other.layer1.copy(),
        );

        let (l2_weights, l2_biases) = Rocket::cross_over_layer(
            self.layer2.copy(),
            other.layer2.copy(),
        );

        let (l3_weights, l3_biases) = Rocket::cross_over_layer(
            self.layer3.copy(),
            other.layer3.copy(),
        );

        let (l4_weights, l4_biases) = Rocket::cross_over_layer(
            self.layer4.copy(),
            other.layer4.copy(),
        );

        Rocket {

            pos: {
                vec2(0.0, -(height as f32) / 2.0)
            },

            vel: {
                vec2(0.0, 0.0)
            },

            acc: {
                vec2(0.0, 0.0)
            },

            layer1: Layer::new_with_values(
                self.layer1.n_inputs,
                self.layer1.n_neurons,
                l1_weights,
                l1_biases
            ),

            layer2: Layer::new_with_values(
                self.layer2.n_inputs,
                self.layer2.n_neurons,
                l2_weights,
                l2_biases
            ),

            layer3: Layer::new_with_values(
                self.layer3.n_inputs,
                self.layer3.n_neurons,
                l3_weights,
                l3_biases
            ),

            layer4: Layer::new_with_values(
                self.layer4.n_inputs,
                self.layer4.n_neurons,
                l4_weights,
                l4_biases
            ),

            target: self.target,

            color: self.color,
        }
    }

    pub fn update(&mut self) {
        self.layer1.forward(
            &vec![self.pos.x.into(), self.pos.y.into(), self.target.x.into(), self.target.y.into()]
        );
        self.layer1.relu();

        self.layer2.forward(
            &vec![
                self.layer1.outputs[0],
                self.layer1.outputs[1],
                self.layer1.outputs[2],
            ]
        );
        self.layer2.relu();

        self.layer3.forward(
            &vec![
                self.layer2.outputs[0],
                self.layer2.outputs[1],
                self.layer2.outputs[2],
            ]
        );
        self.layer3.relu();

        self.layer4.forward(
            &vec![
                self.layer3.outputs[0],
                self.layer3.outputs[1],
                self.layer3.outputs[2],
            ]
        );
        self.layer4.softmax();

        // get the index of the maximum output of layer4
        let mut max_index = 0;
        let mut max_value = self.layer4.outputs[0];
        for i in 0..self.layer4.outputs.len() {
            if self.layer4.outputs[i] > max_value {
                max_index = i;
                max_value = self.layer4.outputs[i];
            }
        }

        match max_index {
            0 => {
                self.vel.y += 0.1;
            },
            1 => {
                self.vel.x += 0.1;
            },
            2 => {
                self.vel.x -= 0.1;
            },
            _ => {
                println!("Error: max_index is not 0, 1, or 2");
            }
        }

        self.vel += self.acc;
        self.pos += self.vel;
        self.acc *= 0.0;
    }

    pub fn display(&self, draw: &Draw) {
        draw.rect()
            .x_y(self.pos.x, self.pos.y)
            .w_h(5.0, 25.0)
            .rotate(
                {
                    if self.vel.x == 0.0 && self.vel.y == 0.0 {
                        0.0
                    } else {
                        let v = vec2(0.0, 1.0);
                        self.vel.angle() + v.angle()
                    }
                }
            )
            .color(self.color);
    }
}

pub trait Evaluation {
    fn evaluation(&self, width: u32) -> Vec<Rocket>;
    fn selection(&self, mating_pool: &Vec::<Rocket>, population_size: u32, height: u32) -> Vec<Rocket>;
}

impl Evaluation for Vec<Rocket> {
    fn evaluation(&self, width: u32) -> Vec<Rocket> {
        let mut mating_pool = Vec::<Rocket>::new();
        for rocket in self {
            let times_to_populate = rocket.evaluate(width);
            for _ in 0..times_to_populate {
                mating_pool.push(rocket.copy());
            }
        }
        mating_pool
    }

    fn selection(&self, mating_pool: &Vec::<Rocket>, population_size: u32, height: u32) -> Vec<Rocket> {
        let mut new_population = Vec::<Rocket>::new();
        for _ in 0..population_size {
            let parent_a = mating_pool.choose(&mut rand::thread_rng()).unwrap();
            let parent_b = mating_pool.choose(&mut rand::thread_rng()).unwrap();
            new_population.push(parent_a.cross_over(parent_b, height));
        }
        new_population
    }
}