use minitorch_rust::core::nn::module::Module;
use minitorch_rust::core::tensor::AbstractTensor;
use minitorch_rust::models::XORClassifier;

use rand::SeedableRng;
use rand::rngs::StdRng;

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
