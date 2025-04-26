# TODO List for Greenblatt ARC Demo

## Testing and Performance Optimization

### Sandbox Execution Robustness (Priority #1)
- [x] Improve sandbox execution robustness
  - Ensure that a single problematic program cannot cause the entire batch to fail. This is best done by removing the looping and calling run_python... for each program (using the same session for the same batch).
  - Implement graceful failure handling for individual programs without affecting the entire batch
  - Add per-program timeouts within the batch execution loop
  - Remove all other timeouts because I don't understand their purpose.

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
