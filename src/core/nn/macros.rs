#[macro_export]
macro_rules! modules {
    ($model:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        pub struct $model {
            $(pub $field: $ty,)*
        }

        impl $crate::core::nn::module::Module for $model {
            fn for_each_own_module(
                &self,
                f: &mut dyn FnMut(&str, &dyn $crate::core::nn::module::Module),
            ) {
                $(f(stringify!($field), &self.$field);)*
            }

            fn for_each_own_module_mut(
                &mut self,
                f: &mut dyn FnMut(&str, &mut dyn $crate::core::nn::module::Module),
            ) {
                $(f(stringify!($field), &mut self.$field);)*
            }

            fn module(&self, name: &str) -> Option<&dyn $crate::core::nn::module::Module> {
                match name {
                    $(stringify!($field) => Some(&self.$field),)*
                    _ => None,
                }
            }

            fn module_mut(
                &mut self,
                name: &str,
            ) -> Option<&mut dyn $crate::core::nn::module::Module> {
                match name {
                    $(stringify!($field) => Some(&mut self.$field),)*
                    _ => None,
                }
            }
        }
    };
}
