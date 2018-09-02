/*
Copyright 2018 Gordon Brown

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __SEQUENTIAL_TRANSFORM_REDUCE_H__
#define __SEQUENTIAL_TRANSFORM_REDUCE_H__

#include <bits/policies.h>

namespace cppcon {

template <class ForwardIt, class T, class UnaryOperation, class BinaryOperation>
T transform_reduce(seq_execution_policy_t policy, ForwardIt first,
                   ForwardIt last, T init, UnaryOperation unary_op,
                   BinaryOperation binary_op) {
  for (; first != last; ++first) {
    init = binary_op(init, unary_op(*first));
  }
  return init;
}

}  // namespace cppcon

#endif  // __SEQUENTIAL_TRANSFORM_REDUCE_H__