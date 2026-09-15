use minitorch_rust::core::GraphTensor;
use minitorch_rust::core::dtype::DtypeStyler;
use minitorch_rust::core::tensor::AbstractTensor;

fn assert_shape<T: DtypeStyler>(t: &GraphTensor<T>, expected: &[usize]) {
    assert_eq!(t.shape(), expected, "shape mismatch for {t:?}");
}

#[test]
fn int_add_mul_and_broadcast() {
    let a = GraphTensor::<i32>::wrap(vec![1, 2, 3], false);
    let b = GraphTensor::<i32>::wrap(10, false);

    let sum = &a + &b;
    assert_shape(&sum, &[3]);
    assert_eq!(*sum.at(&[0]), 11);
    assert_eq!(*sum.at(&[2]), 13);

    let prod = &a * &b;
    assert_shape(&prod, &[3]);
    assert_eq!(*prod.at(&[1]), 20);

    let scaled = &a * 2;
    assert_eq!(*scaled.at(&[2]), 6);

    let shifted = &a + 1;
    assert_eq!(*shifted.at(&[0]), 2);

    let mx = GraphTensor::<i32>::maximum(&a, &b);
    assert_eq!(*mx.at(&[0]), 10);
}

#[test]
fn int_comparison_masks() {
    let a = GraphTensor::<i32>::wrap(vec![1, 5, 3, 8], false);
    let b = GraphTensor::<i32>::wrap(4, false);

    assert_eq!(*a.gt(&b).at(&[0]), 0);
    assert_eq!(*a.gt(&b).at(&[1]), 1);
    assert_eq!(*a.gte(&b).at(&[2]), 0);
    assert_eq!(*a.lt(&b).at(&[3]), 0);
    assert_eq!(*a.lte(&b).at(&[3]), 0);
    assert_eq!(*a.lte(&b).at(&[0]), 1);
}

#[test]
fn int_reductions() {
    let a = GraphTensor::<i64>::wrap(vec![vec![1, 2], vec![3, 4]], false);

    let s = a.sum(&[0], true);
    assert_eq!(*s.at(&[0, 0]), 4);
    assert_eq!(*s.at(&[0, 1]), 6);

    let m = a.max(&[1], true);
    assert_eq!(*m.at(&[0, 0]), 2);
    assert_eq!(*m.at(&[1, 0]), 4);

    let total = a.sum(&[], false);
    assert_eq!(*total.at(&[]), 10);

    let am = a.argmax(0, false);
    assert_eq!(*am.at(&[0]), 1.0);
    assert_eq!(*am.at(&[1]), 1.0);

    let oh = GraphTensor::<i32>::wrap(vec![0, 2, 1], false).one_hot(3);
    assert_shape(&oh, &[3, 3]);
    assert_eq!(*oh.at(&[0, 0]), 1.0);
    assert_eq!(*oh.at(&[0, 2]), 0.0);
    assert_eq!(*oh.at(&[1, 2]), 1.0);
    assert_eq!(*oh.at(&[2, 1]), 1.0);
}

#[test]
fn int_matmul() {
    let a = GraphTensor::<i32>::wrap(vec![vec![1, 2]], false);
    let b = GraphTensor::<i32>::wrap(vec![vec![3], vec![4]], false);

    let r = GraphTensor::matmul(&a, &b);
    assert_shape(&r, &[1, 1]);
    assert_eq!(*r.at(&[0, 0]), 11);
}

#[test]
fn int_structure_ops() {
    let a = GraphTensor::<u8>::wrap(vec![1, 2, 3], false);

    let u = a.unsqueeze(0);
    assert_shape(&u, &[1, 3]);

    let sq = u.squeeze(0);
    assert_shape(&sq, &[3]);

    let cd = a.copy_d();
    assert_eq!(*cd.at(&[2]), 3);

    let s = GraphTensor::stack(&[a.copy_s(), a.copy_s()]);
    assert_shape(&s, &[2, 3]);
    assert_eq!(*s.at(&[1, 1]), 2);
}

#[test]
fn f32_forward_ops() {
    let a = GraphTensor::<f32>::wrap(vec![4.0, 9.0], false);
    let b = GraphTensor::<f32>::wrap(vec![1.0, 3.0], false);

    assert_eq!(*(&a / 2.0_f32).at(&[0]), 2.0);
    assert_eq!(*(&a - &b).at(&[1]), 6.0);
    assert_eq!(*(-&a).at(&[0]), -4.0);
    assert_eq!(*a.sqrt().at(&[1]), 3.0);
    assert_eq!(*a.ln().at(&[0]), 4.0_f32.ln());
    assert_eq!(*a.exp().at(&[0]), 4.0_f32.exp());
    assert!(f32::abs(*a.pow(&b).at(&[0]) - 4.0_f32.powf(1.0)) < 1e-5);
    assert_eq!(*(&b + 1.0_f32).at(&[0]), 2.0);
    assert_eq!(*(&b * 2.0_f32).at(&[1]), 6.0);

    let m = a.mean(&[0], false);
    assert_eq!(*m.at(&[]), 6.5);

    let n = a.norm();
    assert!(f32::abs(*n.at(&[]) - (16.0_f32 + 81.0_f32).sqrt()) < 1e-5);
}

#[test]
fn f32_backward() {
    let a = GraphTensor::<f32>::wrap(vec![2.0, 4.0], true);
    let b = GraphTensor::<f32>::wrap(vec![1.0, 5.0], true);

    let r = (&a * &b).sum(&[], false);
    let grads = r.backward(true);
    let da = grads.get(&a.to_key()).unwrap();
    let db = grads.get(&b.to_key()).unwrap();

    assert_eq!(*da.at(&[0]), 1.0);
    assert_eq!(*da.at(&[1]), 5.0);
    assert_eq!(*db.at(&[0]), 2.0);
    assert_eq!(*db.at(&[1]), 4.0);
}

#[test]
fn bool_logic_ops() {
    let t1 = GraphTensor::<bool>::wrap(vec![true, false, true], false);
    let t2 = GraphTensor::<bool>::wrap(vec![false, false, true], false);

    let land = t1.land(&t2);
    assert_eq!(*land.at(&[0]), false);
    assert_eq!(*land.at(&[2]), true);

    let lor = t1.lor(&t2);
    assert_eq!(*lor.at(&[0]), true);
    assert_eq!(*lor.at(&[1]), false);

    let lnot = t1.lnot();
    assert_eq!(*lnot.at(&[0]), false);
    assert_eq!(*lnot.at(&[1]), true);
}

#[test]
fn copy_d_attaches_grad_edge() {
    let x = GraphTensor::<f64>::wrap(vec![1.0, 2.0, 3.0], true);
    let y = x.copy_d();
    let z = (&y * 2.0_f64).sum(&[], false);
    let grads = z.backward(true);
    let dx = grads.get(&x.to_key()).unwrap();

    assert_eq!(*dx.at(&[0]), 2.0);
    assert_eq!(*dx.at(&[2]), 2.0);
}

#[test]
fn typed_labels_feed_f64_loss_pipeline() {
    let labels = GraphTensor::<i32>::wrap(vec![0, 2, 1], false);
    let logits = GraphTensor::<f64>::wrap(
        vec![vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0], vec![3.0, 2.0, 1.0]],
        true,
    );

    let gts_oh = labels.one_hot(logits.shape()[1]);
    assert_shape(&gts_oh, &[3, 3]);
    assert_eq!(*gts_oh.at(&[0, 0]), 1.0);
    assert_eq!(*gts_oh.at(&[1, 1]), 0.0);
    assert_eq!(*gts_oh.at(&[1, 2]), 1.0);

    let loss = (-&(&gts_oh * &logits)).mean(&[], false);
    assert_shape(&loss, &[]);

    let grads = loss.backward(true);
    let dlogits = grads.get(&logits.to_key()).unwrap();
    assert_eq!(*dlogits.at(&[0, 0]), -1.0 / 9.0);
    assert_eq!(*dlogits.at(&[1, 2]), -1.0 / 9.0);
    assert_eq!(*dlogits.at(&[2, 1]), -1.0 / 9.0);
    assert_eq!(*dlogits.at(&[0, 2]), 0.0);
}

#[test]
fn exact_cast_chain_widens() {
    let a = GraphTensor::<i8>::wrap(vec![1, 2, 3], false);
    let b = a.cast::<i32>();
    let c = b.cast::<i64>();

    assert_eq!(*b.at(&[1]), 2);
    assert_eq!(*c.at(&[2]), 3);

    let f = GraphTensor::<f32>::wrap(vec![0.25], false).cast::<f64>();
    assert_eq!(*f.at(&[0]), 0.25);
}