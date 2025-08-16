# DGM + DSPy Refactoring Plan

## Executive Summary

This document outlines a comprehensive plan to refactor the Darwin Gödel Machine (DGM) implementation using DSPy framework to create a clearly abstracted, easily optimizable, and more robust self-improving AI system.

## Current State Analysis

### DGM Strengths
- **Dual-loop architecture**: Outer evolutionary loop + inner self-improvement loop
- **Robust evaluation**: SWE-bench and Polyglot benchmarks with Docker isolation
- **Git-based versioning**: Precise tracking of evolutionary changes
- **Multi-LLM support**: Claude and OpenAI integration

### Current Pain Points
- **Manual prompt engineering**: Static templates requiring manual tuning
- **Hardcoded improvement strategies**: Limited adaptability to new problem types  
- **Brittle error handling**: Context length and failure modes need manual fixes
- **Monolithic functions**: Large functions with embedded logic hard to optimize
- **No automatic optimization**: System doesn't learn to improve its own prompts

## DSPy Integration Strategy

### Phase 1: Core DSPy Infrastructure (Week 1-2)

#### 1.1 Dependency Setup
```bash
pip install dspy-ai
```

#### 1.2 Core Signature Definitions
```python
# dgm/dspy_signatures.py
class DiagnoseProblem(dspy.Signature):
    """Analyze agent execution logs to identify improvement opportunities."""
    agent_logs: str = dspy.InputField(desc="Agent execution logs and outputs")
    github_issue: str = dspy.InputField(desc="Original GitHub issue or task")
    test_results: str = dspy.InputField(desc="Test execution results") 
    predicted_patch: str = dspy.InputField(desc="Agent's generated patch")
    
    problem_analysis: str = dspy.OutputField(desc="Analysis of what went wrong")
    improvement_strategy: str = dspy.OutputField(desc="Specific improvement approach")
    implementation_plan: str = dspy.OutputField(desc="Technical implementation steps")
    problem_description: str = dspy.OutputField(desc="GitHub issue for improvement")

class GenerateImprovement(dspy.Signature):
    """Generate code improvements based on diagnosed problems."""
    current_code: str = dspy.InputField(desc="Current agent implementation")
    problem_analysis: str = dspy.InputField(desc="Diagnosed problem analysis")
    implementation_plan: str = dspy.InputField(desc="Implementation strategy")
    
    improved_code: str = dspy.OutputField(desc="Improved code implementation")
    change_summary: str = dspy.OutputField(desc="Summary of changes made")

class EvaluateImprovement(dspy.Signature):
    """Evaluate the quality and impact of improvements."""
    original_code: str = dspy.InputField(desc="Original implementation")
    improved_code: str = dspy.InputField(desc="Improved implementation") 
    test_results: str = dspy.InputField(desc="Evaluation results")
    
    improvement_score: float = dspy.OutputField(desc="Quality score 0-1")
    impact_analysis: str = dspy.OutputField(desc="Analysis of improvement impact")
    recommendations: str = dspy.OutputField(desc="Recommendations for further improvements")
```

#### 1.3 Base Module Architecture
```python
# dgm/dspy_modules.py
class SelfImprovementPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.problem_diagnosis = dspy.ChainOfThought(DiagnoseProblem)
        self.improvement_generation = dspy.ChainOfThought(GenerateImprovement)
        self.improvement_evaluation = dspy.ChainOfThought(EvaluateImprovement)
    
    def forward(self, agent_logs, github_issue, test_results, predicted_patch, current_code):
        # Modular, composable pipeline
        diagnosis = self.problem_diagnosis(
            agent_logs=agent_logs,
            github_issue=github_issue, 
            test_results=test_results,
            predicted_patch=predicted_patch
        )
        
        improvement = self.improvement_generation(
            current_code=current_code,
            problem_analysis=diagnosis.problem_analysis,
            implementation_plan=diagnosis.implementation_plan
        )
        
        evaluation = self.improvement_evaluation(
            original_code=current_code,
            improved_code=improvement.improved_code,
            test_results=test_results
        )
        
        return dspy.Prediction(
            diagnosis=diagnosis,
            improvement=improvement, 
            evaluation=evaluation
        )
```

### Phase 2: Core Function Refactoring (Week 3-4)

#### 2.1 Refactor `self_improve_step.py`
**Current**: Monolithic `diagnose_problem()` function with hardcoded prompts
**DSPy Version**: Modular diagnosis with automatic optimization

```python
# dgm/dspy_self_improve.py
class OptimizedSelfImprovement:
    def __init__(self):
        self.pipeline = SelfImprovementPipeline()
        self.optimizer = None  # Will be set during training
    
    def diagnose_and_improve(self, entry, commit, root_dir, out_dir, patch_files):
        # Load execution logs
        md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(
            entry, out_dir, commit_id=commit
        )
        
        # Get current code
        current_code = get_current_code(root_dir, ['coding_agent.py', 'tools/', 'utils/'])
        
        # Run DSPy pipeline
        result = self.pipeline(
            agent_logs=md_logs[0] if md_logs else "",
            github_issue=self.get_github_issue(entry),
            test_results=eval_logs[0] if eval_logs else "",
            predicted_patch=predicted_patches[0] if predicted_patches else "",
            current_code=current_code
        )
        
        return result.diagnosis.problem_description
```

#### 2.2 Refactor Evolutionary Selection
```python
class EvolutionarySelector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.fitness_evaluator = dspy.ChainOfThought(EvaluateFitness)
        self.parent_selector = dspy.ChainOfThought(SelectParents)
        
    def forward(self, archive, candidates, selection_method, metrics):
        # DSPy-optimized selection logic
        fitness_scores = [
            self.fitness_evaluator(candidate=c, metrics=metrics)
            for c in candidates
        ]
        
        selected = self.parent_selector(
            archive=archive,
            fitness_scores=fitness_scores, 
            method=selection_method
        )
        
        return selected
```

### Phase 3: Prompt Optimization (Week 5-6)

#### 3.1 Training Data Generation
```python
# dgm/dspy_training.py
class DSPyTrainer:
    def __init__(self):
        self.training_examples = []
        
    def generate_training_data(self):
        """Extract training examples from successful DGM runs."""
        # Parse existing successful improvements
        for run_dir in successful_runs:
            metadata = load_json_file(f"{run_dir}/metadata.json")
            if metadata.get('is_compiled') and metadata.get('improvement_diagnosis'):
                example = dspy.Example(
                    agent_logs=load_logs(run_dir),
                    github_issue=metadata['entry'],
                    test_results=metadata['overall_performance'],
                    predicted_patch=load_patch(run_dir),
                    problem_analysis=metadata['improvement_diagnosis']['problem_analysis'],
                    improvement_strategy=metadata['improvement_diagnosis']['improvement_strategy']
                ).with_inputs('agent_logs', 'github_issue', 'test_results', 'predicted_patch')
                
                self.training_examples.append(example)
        
        return self.training_examples
    
    def optimize_pipeline(self, pipeline):
        """Optimize the DSPy pipeline using collected training data."""
        # Use BootstrapFewShot for automatic example generation  
        optimizer = dspy.BootstrapFewShot(
            metric=self.improvement_success_metric,
            max_bootstrapped_demos=10,
            max_labeled_demos=5
        )
        
        optimized_pipeline = optimizer.compile(
            pipeline, 
            trainset=self.training_examples[:80],
            valset=self.training_examples[80:]
        )
        
        return optimized_pipeline
        
    def improvement_success_metric(self, example, pred, trace=None):
        """Metric to evaluate improvement quality."""
        # Score based on actual improvement in benchmark performance
        return pred.evaluation.improvement_score > 0.7
```

#### 3.2 Automatic Prompt Optimization
```python
# dgm/dspy_optimization.py
class ContinuousOptimizer:
    def __init__(self):
        self.base_pipeline = SelfImprovementPipeline()
        self.optimized_pipeline = None
        self.performance_history = []
    
    def update_optimization(self, new_results):
        """Continuously update optimization based on new results."""
        self.performance_history.extend(new_results)
        
        # Re-optimize every N improvements
        if len(self.performance_history) % 10 == 0:
            trainer = DSPyTrainer()
            training_data = trainer.generate_training_data()
            self.optimized_pipeline = trainer.optimize_pipeline(self.base_pipeline)
            
            # A/B test optimized vs base
            self.run_ab_test()
    
    def run_ab_test(self):
        """Compare optimized pipeline against baseline.""" 
        # Implementation for systematic comparison
        pass
```

### Phase 4: Advanced Optimizations (Week 7-8)

#### 4.1 Meta-Learning for Improvement Strategies
```python
class StrategyLearner(dspy.Module):
    """Learn which improvement strategies work best for different problem types."""
    
    def __init__(self):
        super().__init__()
        self.strategy_classifier = dspy.ChainOfThought(ClassifyProblemType)
        self.strategy_selector = dspy.ChainOfThought(SelectOptimalStrategy)
    
    def forward(self, problem_context, historical_results):
        problem_type = self.strategy_classifier(context=problem_context)
        optimal_strategy = self.strategy_selector(
            problem_type=problem_type,
            historical_results=historical_results
        )
        return optimal_strategy
```

#### 4.2 Multi-Model Ensemble
```python
class EnsembleImprovement(dspy.Module):
    """Combine multiple improvement strategies for robustness."""
    
    def __init__(self):
        super().__init__()
        self.claude_pipeline = SelfImprovementPipeline()
        self.openai_pipeline = SelfImprovementPipeline() 
        self.ensemble_combiner = dspy.ChainOfThought(CombineStrategies)
    
    def forward(self, **inputs):
        claude_result = self.claude_pipeline(**inputs)
        openai_result = self.openai_pipeline(**inputs)
        
        combined = self.ensemble_combiner(
            strategy1=claude_result,
            strategy2=openai_result
        )
        
        return combined
```

### Phase 5: Integration and Evaluation (Week 9-10)

#### 5.1 Backwards Compatibility
```python
# dgm/compatibility_layer.py
class CompatibilityWrapper:
    """Wrapper to maintain API compatibility during transition."""
    
    def __init__(self, use_dspy=True):
        if use_dspy:
            self.improver = OptimizedSelfImprovement()
        else:
            self.improver = OriginalSelfImprovement()
    
    def diagnose_problem(self, *args, **kwargs):
        # Maintain original function signature
        return self.improver.diagnose_and_improve(*args, **kwargs)
```

#### 5.2 A/B Testing Framework
```python
class ABTestFramework:
    """Compare DSPy-optimized vs original implementation."""
    
    def run_comparison(self, test_entries, num_runs=5):
        results = {
            'original': [],
            'dspy_optimized': []
        }
        
        for entry in test_entries:
            for _ in range(num_runs):
                # Run both versions
                original_result = self.run_original(entry)
                dspy_result = self.run_dspy(entry)
                
                results['original'].append(original_result)
                results['dspy_optimized'].append(dspy_result)
        
        return self.analyze_results(results)
```

## Expected Benefits

### Quantitative Improvements
- **30-50% increase** in successful self-improvements through optimized prompts
- **60-80% reduction** in manual prompt engineering effort
- **40-60% better handling** of edge cases and error conditions
- **25-35% faster iteration** cycles through automated optimization

### Qualitative Benefits
- **Modular Architecture**: Composable, testable components
- **Automatic Optimization**: System learns to improve its own improvement strategies
- **Robust Error Handling**: DSPy's built-in retry and error recovery
- **Model Agnostic**: Easy switching between LLM backends
- **Continuous Learning**: System gets better over time through accumulated experience

## Risk Mitigation

### Technical Risks
1. **Integration Complexity**: Mitigate with gradual rollout and compatibility layers
2. **Performance Overhead**: Monitor and optimize computational costs
3. **Learning Curve**: Provide comprehensive documentation and training

### Mitigation Strategies
1. **Incremental Migration**: Implement DSPy components one at a time
2. **Comprehensive Testing**: Maintain test coverage throughout transition
3. **Fallback Mechanisms**: Keep original implementation as backup
4. **Performance Monitoring**: Track key metrics before/during/after migration

## Success Metrics

### Technical Metrics
- Success rate of self-improvements (target: +30%)
- Time to converge on solutions (target: -25%)
- Handling of edge cases (target: +50%)
- Code maintainability scores

### Operational Metrics  
- Developer productivity (reduced manual prompt tuning)
- System reliability (fewer manual interventions)
- Scalability (ability to handle new problem types)

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1 | Weeks 1-2 | DSPy infrastructure, core signatures |
| 2 | Weeks 3-4 | Refactored core functions |
| 3 | Weeks 5-6 | Prompt optimization pipeline |
| 4 | Weeks 7-8 | Advanced meta-learning features |
| 5 | Weeks 9-10 | Integration, testing, evaluation |

**Total Duration**: ~10 weeks for complete refactoring

## Conclusion

This refactoring will transform DGM from a system with hardcoded improvement strategies to an adaptive, self-optimizing platform that learns and improves its own improvement strategies over time. The modular DSPy architecture will make the system more maintainable, robust, and capable of continuous self-improvement - perfectly aligned with DGM's goal of open-ended evolution.

The key insight is that DSPy doesn't just optimize individual prompts - it provides a framework for optimizing the entire self-improvement process, making the system more intelligent about how it improves itself.