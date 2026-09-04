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

impl Module for XORClassifier {}

// class CovertypeClassifier: public nn::Module, nn::Forward1 {
//     public:
//         nn::Linear lin1;
//         nn::ReLU relu;
//         nn::Linear lin2;
//         nn::Linear lin3;

//         CovertypeClassifier():
//             lin1(54, 100),
//             relu{},
//             lin2(100, 100),
//             lin3(100, 7) {
//                 register_module("lin1", lin1);
//                 register_module("lin2", lin2);
//                 register_module("lin3", lin3);
//                 reset_parameters();
//             }
//         void reset_parameters() {
//             nn::kaiming_uniform_inplace(lin1.m_weight, get_rng());
//             lin1.m_bias.fill_inplace(0.0f);
//             nn::kaiming_uniform_inplace(lin2.m_weight, get_rng());
//             lin2.m_bias.fill_inplace(0.0f);
//             nn::kaiming_uniform_inplace(lin3.m_weight, get_rng());
//             lin3.m_bias.fill_inplace(0.0f);
//         }

//         Tensor evaluate(
//             mt::data::DataLoader<Tensor, Tensor>& dl,
//             nn::Loss& criterion
//         ) {
//             float curr_loss = 0;
//             float curr_sample_count = 0;

//             for (size_t step { 0 }; step < dl.size(); ++step) {
//                 auto [inputs, gts] = dl.get_batch(step);

//                 Tensor prs_oh = forward(inputs);

//                 Tensor gts_oh = gts.one_hot(prs_oh.shape()[1]);

//                 Tensor loss = criterion.forward(prs_oh, gts_oh);

//                 curr_loss += loss.item()*(static_cast<float>(inputs.shape()[0]));
//                 curr_sample_count += static_cast<float>(inputs.shape()[0]);
//             }
//             float total_loss = curr_loss / curr_sample_count;
//             return Tensor({}, total_loss);
//         }
//     };
