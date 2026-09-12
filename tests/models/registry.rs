use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::nn::compute::Linear;
use minitorch_rust::core::nn::module::Module;
use minitorch_rust::core::tensor::AbstractTensor;
use minitorch_rust::models::XORClassifier;
use minitorch_rust::module;

use rand::SeedableRng;
use rand::rngs::StdRng;

module! {
    TestModel {
        modules {
            lin: Linear,
        },
        params {
            pos_embed,
            scale,
        }
    }
}

fn test_model() -> TestModel {
    TestModel {
        lin: Linear::new(2, 3, true, StdRng::seed_from_u64(42)),
        pos_embed: GraphTensor::new(vec![4], 0.5, true),
        scale: GraphTensor::new(vec![], 1.0, false),
    }
}

#[test]
fn module_lookup_by_name_and_path() {
    let model = XORClassifier::new(StdRng::seed_from_u64(42));

    assert!(model.module("lin1").is_some());
    assert!(model.module("relu").is_some());
    assert!(model.module("lin3").is_some());
    assert!(model.module("nope").is_none());

    assert!(model.module_path("lin1").is_some());
    assert!(model.module_path("lin2").is_some());
    assert!(model.module_path("nope.deeper").is_none());
}

#[test]
fn module_dynamic_access_reaches_the_same_parameters() {
    let model = XORClassifier::new(StdRng::seed_from_u64(42));

    let lin1 = model.module("lin1").unwrap();
    let mut params: Vec<String> = Vec::new();
    lin1.for_each_param(&mut |name, _| params.push(name.to_string()));
    assert_eq!(params, vec!["weight".to_string(), "bias".to_string()]);
}

#[test]
fn module_mut_targets_only_the_requested_subtree() {
    let mut model = XORClassifier::new(StdRng::seed_from_u64(42));

    model
        .module_mut("lin3")
        .unwrap()
        .set_requires_grad(false, true);

    let mut lin3_grad_off = true;
    model
        .lin3
        .for_each_param_mut(&mut |_, p| lin3_grad_off &= !p.requires_grad());
    assert!(lin3_grad_off);

    let mut lin1_grad_on = true;
    model
        .lin1
        .for_each_param_mut(&mut |_, p| lin1_grad_on &= p.requires_grad());
    assert!(lin1_grad_on);
}

#[test]
fn param_lookup_by_name_and_path() {
    let model = test_model();

    assert!(model.param("pos_embed").is_some());
    assert!(model.param("scale").is_some());
    assert!(model.param("nope").is_none());

    assert!(model.param_path("pos_embed").is_some());
    assert!(model.param_path("lin.weight").is_some());
    assert!(model.param_path("lin.bias").is_some());
    assert!(model.param_path("nope.deeper").is_none());
}

#[test]
fn param_mut_targets_only_the_requested_param() {
    let mut model = test_model();

    model.param_mut("scale").unwrap().set_requires_grad(false);

    assert!(!model.scale.requires_grad());
    assert!(model.pos_embed.requires_grad());
    assert!(model.param_path_mut("scale").unwrap().requires_grad() == false);
    assert!(model.param_path_mut("lin.weight").unwrap().requires_grad());
    assert!(model.param_path_mut("lin.bias").unwrap().requires_grad());
}

#[test]
fn recursive_param_enumeration_visits_own_and_child_params() {
    let model = test_model();

    let mut params: Vec<String> = Vec::new();
    model.for_each_param(&mut |name, _| params.push(name.to_string()));
    params.sort();

    assert_eq!(
        params,
        vec![
            "lin.bias".to_string(),
            "lin.weight".to_string(),
            "pos_embed".to_string(),
            "scale".to_string(),
        ]
    );
}
