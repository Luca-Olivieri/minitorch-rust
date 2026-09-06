use std::collections::HashMap;

use rand::{SeedableRng, rngs::StdRng};

use crate::core::{GraphTensor, nn::{activate::ReLU, compute::Linear, module::{Forward1, Module}}};

pub struct XORClassifier {
    pub lin1: Linear,
    pub relu: ReLU,
    pub lin2: Linear,
    pub lin3: Linear
}

impl XORClassifier {

    pub fn new(
        mut rng: StdRng
    ) -> Self {

        let lin1 = Linear::new(2, 100, true, StdRng::from_rng(&mut rng)); // TODO how to clone an rng properly
        let relu = ReLU::new();
        let lin2 = Linear::new(100, 100, true, StdRng::from_rng(&mut rng));
        let lin3 = Linear::new(100, 2, true, StdRng::from_rng(&mut rng));

        Self {lin1, relu, lin2, lin3 }
    }
}

impl Forward1 for XORClassifier {
    fn forward(
        &self,
        input: &GraphTensor
    ) -> GraphTensor {
        let y1 = self.lin1.forward(&input);
        let y2 = self.relu.forward(&y1);
        let y3 = self.lin2.forward(&y2);
        let y4 = self.relu.forward(&y3);
        let logits = self.lin3.forward(&y4);
        logits
    }
}

impl Module for XORClassifier {
    fn modules(
        &self
    ) -> HashMap<String, & dyn Module> {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("linear_1"), &self.lin1 as &dyn Module);
        out_map.insert(String::from("relu"), &self.relu as &dyn Module);
        out_map.insert(String::from("linear_2"), &self.lin2 as &dyn Module);
        out_map.insert(String::from("linear_3"), &self.lin3 as &dyn Module);

        out_map
    }

    fn modules_mut(
        &mut self
    ) -> HashMap<String, &mut dyn Module> {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("linear_1"), &mut self.lin1 as &mut dyn Module);
        out_map.insert(String::from("relu"), &mut self.relu as &mut dyn Module);
        out_map.insert(String::from("linear_2"), &mut self.lin2 as &mut dyn Module);
        out_map.insert(String::from("linear_3"), &mut self.lin3 as &mut dyn Module);

        out_map
    }

    fn parts_mut(&mut self) -> (HashMap<String, &mut GraphTensor>, HashMap<String, &mut dyn Module>) {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("linear_1"), &mut self.lin1 as &mut dyn Module);
        out_map.insert(String::from("relu"), &mut self.relu as &mut dyn Module);
        out_map.insert(String::from("linear_2"), &mut self.lin2 as &mut dyn Module);
        out_map.insert(String::from("linear_3"), &mut self.lin3 as &mut dyn Module);

        (HashMap::new(), out_map)
    }
}

pub struct CovertypeClassifier {
    pub lin1: Linear,
    pub relu: ReLU,
    pub lin2: Linear,
    pub lin3: Linear
}

impl CovertypeClassifier {

    pub fn new(
        mut rng: StdRng
    ) -> Self {

        let lin1 = Linear::new(54, 100, true, StdRng::from_rng(&mut rng)); // TODO how to clone an rng properly
        let relu = ReLU::new();
        let lin2 = Linear::new(100, 100, true, StdRng::from_rng(&mut rng));
        let lin3 = Linear::new(100, 7, true, StdRng::from_rng(&mut rng));

        Self {lin1, relu, lin2, lin3 }
    }
}

impl Forward1 for CovertypeClassifier {
    fn forward(
        &self,
        input: &GraphTensor
    ) -> GraphTensor {
        let y1 = self.lin1.forward(&input);
        let y2 = self.relu.forward(&y1);
        let y3 = self.lin2.forward(&y2);
        let y4 = self.relu.forward(&y3);
        let logits = self.lin3.forward(&y4);
        logits
    }
}

impl Module for CovertypeClassifier {
    fn modules(
        &self
    ) -> HashMap<String, & dyn Module> {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("linear_1"), &self.lin1 as &dyn Module);
        out_map.insert(String::from("relu"), &self.relu as &dyn Module);
        out_map.insert(String::from("linear_2"), &self.lin2 as &dyn Module);
        out_map.insert(String::from("linear_3"), &self.lin3 as &dyn Module);

        out_map
    }

    fn modules_mut(
        &mut self
    ) -> HashMap<String, &mut dyn Module> {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("linear_1"), &mut self.lin1 as &mut dyn Module);
        out_map.insert(String::from("relu"), &mut self.relu as &mut dyn Module);
        out_map.insert(String::from("linear_2"), &mut self.lin2 as &mut dyn Module);
        out_map.insert(String::from("linear_3"), &mut self.lin3 as &mut dyn Module);

        out_map
    }

    fn parts_mut(&mut self) -> (HashMap<String, &mut GraphTensor>, HashMap<String, &mut dyn Module>) {
        let mut out_map = HashMap::new();
        out_map.insert(String::from("linear_1"), &mut self.lin1 as &mut dyn Module);
        out_map.insert(String::from("relu"), &mut self.relu as &mut dyn Module);
        out_map.insert(String::from("linear_2"), &mut self.lin2 as &mut dyn Module);
        out_map.insert(String::from("linear_3"), &mut self.lin3 as &mut dyn Module);

        (HashMap::new(), out_map)
    }
}
