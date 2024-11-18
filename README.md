# Optimizing GPT-2 on CS2 Using CSL

## Introduction

This project focuses on leveraging the **Cerebras Software Language (CSL)** to directly implement **GPT-2** on the **Cerebras CS2 system**. Our goal is to bypass the conventional graph compiler and instead develop a fine-tuned, custom implementation to achieve **extreme performance control**. By programming in CSL, we aim to unlock the full potential of the **750,000 processing elements (PEs)** in the CS2, maximizing efficiency and customizability for the GPT-2 model.

## Motivation

The Cerebras CS2 system, with its **wafer-scale architecture** and massive parallelism, provides an exceptional opportunity to improve the performance of large-scale language models like GPT-2. By using CSL directly, we can:

- **Gain fine-grained control** over resource allocation and PE communication.
- **Optimize parallel data flows** and computation-specific tasks, rather than relying on generic graph compilation.
- **Adapt the architecture** of GPT-2 to better match the capabilities of the wafer-scale engine, including optimizing for memory locality, data movement, and computational efficiency.

## Research Objectives

1. **CSL Implementation of GPT-2 Building Blocks**: 
   - Develop CSL-based kernels for core operations of GPT-2, including matrix multiplications, self-attention, layer normalization, and feed-forward layers.
   - Explore customized implementation strategies for each layer to take advantage of the massive number of PEs available.

2. **Memory and Data Movement Optimization**: 
   - Design an optimal memory access pattern for the GPT-2 layers to reduce latency.
   - Investigate the use of **non-blocking communication** and **column-major data assignment** to maximize throughput.

3. **Parallelization and Task Distribution**: 
   - Formulate an efficient **PE allocation scheme** to parallelize both computation and communication steps for each transformer block.
   - Use **broadcast-based systolic array communication** to accelerate matrix operations, inspired by systolic array techniques.

4. **Performance Metrics and Benchmarking**: 
   - Establish metrics to measure performance improvements over the standard graph-compiled version.
   - Compare **training throughput**, **latency**, and **memory utilization** of the CSL version against the original.

## Methodology

1. **Understanding Cerebras CS2 Hardware**: 
   - Study the architecture of CS2, including the structure and properties of the **processing elements**, **interconnect network**, and memory hierarchy.
   - Analyze the **Cerebras SDK 1.2** to understand features that can be directly leveraged in CSL.

2. **Mapping GPT-2 Architecture to CSL**:
   - Decompose GPT-2 into its key components and identify the mapping of individual computational elements to CSL operations.
   - Create detailed CSL functions to replace core PyTorch/TensorFlow operations.

3. **Kernel Design and Implementation**:
   - Implement core kernels (e.g., matrix multiplication, self-attention) using **CSL kernels**.
   - Optimize these kernels by minimizing PE idle time and maximizing data reuse.

4. **Testing and Benchmarking**:
   - Verify the correctness of CSL-based implementations through unit testing.
   - Deploy the model on CS2 and benchmark the CSL implementation against a baseline model trained using the graph compiler.

## Challenges and Considerations

- **Memory Constraints**: Efficient memory allocation and reuse will be critical, given the limited on-chip memory per PE.
- **Load Balancing**: Achieving even load distribution across all PEs will require careful tuning, especially for attention layers, which have non-uniform computation patterns.
- **Communication Overhead**: Minimizing the overhead associated with data transfer between PEs is crucial, especially when dealing with self-attention mechanisms that require large-scale broadcasting.

## Deliverables

1. **CSL Kernel Library for GPT-2**: A set of reusable CSL kernels optimized for GPT-2 operations.
2. **Full CSL Implementation of GPT-2**: The complete CSL code for GPT-2, along with scripts for deployment and testing on the CS2 system.
3. **Performance Report**: A detailed analysis of the performance gains achieved compared to the graph-compiled version, including key metrics like throughput, latency, and efficiency.

## Timeline

- **Month 1**: Research CS2 architecture and CSL capabilities; Decompose GPT-2 into core components.
- **Month 2**: Develop CSL kernels for matrix multiplication and layer normalization; Begin PE allocation design.
- **Month 3**: Implement self-attention and feed-forward layers; Focus on optimizing data movement.
- **Month 4**: Integrate all components and perform initial testing on CS2; Benchmark performance.
- **Month 5**: Final optimization, troubleshooting, and generation of performance reports.

## How to Get Started

1. **Setup**: Clone the repository and ensure you have access to the Cerebras CS2 environment.
2. **Documentation**: Refer to the CSL documentation available in `/shared/data1/Projects/Cerebras/docs` for understanding the API and PE operations.
3. **Collaboration**: Feel free to submit pull requests or open issues for discussions regarding implementation strategies.

## Contact

For any questions or suggestions, please reach out to the project maintainer at **[your-email]**.
