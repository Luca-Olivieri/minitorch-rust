pub struct SimpleExpSmoothing {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl SimpleExpSmoothing {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            value: 0.0,
            initialized: false,
        }
    }

    pub fn update_and_get(&mut self, observation: f64) -> f64 {
        if !self.initialized {
            self.value = observation;
            self.initialized = true;
        } else {
            self.value = self.alpha * observation + (1.0 - self.alpha) * self.value;
        }
        self.value
    }

    pub fn update(&mut self, observation: f64) {
        if !self.initialized {
            self.value = observation;
            self.initialized = true;
        } else {
            self.value = self.alpha * observation + (1.0 - self.alpha) * self.value;
        }
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }
}
