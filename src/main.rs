mod rocket;
mod layer;

use nannou::{prelude::*, color::Alpha};
use rocket::Rocket;
use rocket::Evaluation;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const POPULATION_SIZE: u32 = 50;
const TARGET: (f64, f64) = (0.0, 300.0);


struct Model {
    _window: window::Id,
    bg_color: Alpha<Rgb<f64>, f64>,
    rockets: Vec<Rocket>,
    lifetime: u16,
}

fn main() {
    nannou::app(model)
        .event(event)
        .update(update)
        .run();
}

fn model(app: &App) -> Model {
    Model {

        _window: {
            app
                .new_window()
                .view(view)
                .size(WIDTH, HEIGHT)
                .build().
                unwrap()
        },

        bg_color: {
            nannou::color::rgba(
                35.0 / 255.0,
                39.0 / 255.0,
                46.0 / 255.0,
                1.0
            )
        },

        rockets: {
            let mut rockets = Vec::<Rocket>::new();
            for _ in 0..POPULATION_SIZE {
                rockets.push(Rocket::new(WIDTH, HEIGHT, vec2(TARGET.0 as f32, TARGET.1 as f32)));
            }
            rockets
        },

        lifetime: {
            0
        }

    }
}

fn event(_app: &App, _model: &mut Model, _event: Event) {}

fn update(_app: &App, model: &mut Model, _update: Update) {
    model.lifetime += 1;
    model.rockets.iter_mut().for_each(|rocket| rocket.update());

    if model.lifetime >= 200 {
        println!("Reached end of lifetime!");
        model.rockets = {
            model.rockets.selection(
                &model.rockets.evaluation(WIDTH),
                POPULATION_SIZE,
                HEIGHT,
            )
        };
        println!("Reset rockets with new stuff.");
        model.lifetime = 0;
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(model.bg_color);

    draw.ellipse()
        .x_y(TARGET.0 as f32, TARGET.1 as f32)
        .radius(10.0)
        .color(
            nannou::color::rgb(
                218.0 / 255.0,
                100.0 / 255.0,
                110.0 / 255.0,
            )
        );

    model.rockets.iter().for_each(|rocket| rocket.display(&draw));

    draw.to_frame(app, &frame).unwrap();
}