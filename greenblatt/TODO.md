# TODO List for Greenblatt ARC Demo

## Testing and Performance Optimization

### Sandbox Execution Robustness (Priority #1)
- [ ] Improve sandbox execution robustness
  - Add per-program timeouts within the batch execution loop
  - Implement graceful failure handling for individual programs without affecting the entire batch
  - Ensure that a single problematic program cannot cause the entire batch to fail

### Timeout Handling
- [ ] Improve timeout diagnostics
  - Add progress tracking during program generation
  - Implement periodic status updates for long-running operations
  - Create a mechanism to capture and report the current state when a timeout occurs
  - Allow tasks to gracefully report their state before being terminated

### Unit Testing
- [ ] Write unit tests for the evaluation step
  - Test with different program sizes and complexities
  - Test with edge cases (empty grids, large grids)
  - Test timeout handling and error recovery
  - Verify result caching works correctly

### Pressure Testing
- [ ] Pressure test the evaluation with larger batch sizes
  - Test with 64, 128, and 256 programs per batch
  - Monitor memory usage during large batch executions
  - Identify potential bottlenecks in the sandbox execution
  - Test with complex programs that might stress the execution environment

## Code Cleanup
- [ ] Remove debugging code once issues are resolved
  - Clean up the debug_task function
  - Remove excessive logging statements
  - Consolidate timeout parameters to a configuration file
  - Document the timeout strategy in the README

## Documentation
- [ ] Update documentation with performance considerations
  - Document the optimal batch sizes for different scenarios
  - Provide guidelines for timeout settings
  - Explain the caching mechanism and how to use it effectively
