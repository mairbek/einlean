# Future Ideas

## Compile-time size checking for tensor literals

Use a custom macro like `t![[1,2,3],[4,5,6]]` that parses nested list literals, checks that dimensions match the expected tensor shape at elaboration time, and produces a `Tensor.ofData` call. Gives both nice syntax and compile-time safety.
