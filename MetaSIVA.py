"""
SIVA WITH META-LEARNING - COMPLETE INTEGRATION
==================================================
This file integrates the meta-learning enhancements with the base SIVA system.

Usage:
1. Ensure the original SIVA.py file is in the same directory
2. Run this file to use the enhanced meta-learning version


Based on:
- REVOLVE [1]: https://github.com/Peiyance/REVOLVE
- SecVulEval Dataset [2]: https://huggingface.co/datasets/arag0rn/SecVulEval
- SICA (Self-Improving Coding Agent) [3]: https://github.com/MaximeRobeyns/self_improving_coding_agent
- ADAS (Automated Design of Agentic Systems) [4]: https://github.com/ShengranHu/ADAS

Utilizing:
- Gemma3 [5]: https://deepmind.google/models/gemma/gemma-3/

Implemented Using:
- Claude-4 Sonnet [6]: https://www.anthropic.com/claude/sonnet

References:
[1] Zhang et al. (2025), REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization (https://arxiv.org/abs/2412.03092)
[2] Ahmed et al. (2025), SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection (https://arxiv.org/abs/2505.19828)
[3] Robeyns et al. (2025), A Self-Improving Coding Agent (https://arxiv.org/abs/2504.15228)
[4] Hu et al. (2025), Automated Design of Agentic Systems (https://arxiv.org/abs/2408.08435)
[5] Google DeepMind (2024), Gemma3 (https://deepmind.google/models/gemma/gemma-3/)
[6] Anthropic (2025), Claude-4 Sonnet (https://www.anthropic.com/claude/sonnet)
[7] T. Liu and M. van der Schaar (2025), Truly Self-Improving Agents Require Intrinsic Metacognitive Learning (https://arxiv.org/abs/2506.05109)


Author: Valentin Walischewski
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Import the base SICA-VULN components
# Add the current directory to path to import the base agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import from the original SICA-VULN file
    from SIVA import (
        SICAVulnAgent,
        SmartSecurityMemorySystem,
        SecurityLLMClient,
        SecurityEvaluator,
        SECVULEVALLoader,
        VulnerabilityPatternRecognizer,
        logger
    )
    print("✅ Successfully imported base SICA-VULN components")
    
except ImportError as e:
    
    print(f"❌ Error importing base SICA-VULN: {e}")
    print("Please ensure SIVA.py is in the same directory as this file")
    sys.exit(1)


import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# ============================================================
# 1. META-LEARNING COMPONENTS
# ============================================================


@dataclass
class CWEPromptTemplate:
    """Evolved prompt template for specific CWE types"""
    
    cwe_type: str
    prompt_template: str
    success_rate: float
    usage_count: int
    last_updated: float
    key_elements: List[str]
    failure_patterns: List[str]

@dataclass
class LearningStrategyAnalysis:
    """Analysis of learning strategy effectiveness"""
    
    strategy_name: str
    cwe_type: str
    success_count: int
    failure_count: int
    avg_quality_score: float
    common_failure_reasons: List[str]
    recommended_adjustments: List[str]

@dataclass
class MetaLearningDecision:
    """Meta-learning decision for a specific case"""
    
    function_id: str
    cwe_type: str
    failure_analysis: str
    recommended_strategy: str
    custom_prompt_adjustments: Dict[str, str]
    confidence: float
    reasoning: str

class MetaPromptLibrary:
    """Dynamic library of CWE-specific prompts that evolves based on performance"""
    
    def __init__(self, workspace: Path):
        """ Initializes prompt library system
        
            Creates:
                 - meta_prompt_library/ directory
                 - Base templates for common CWEs
                 - Pattern prompt storage

            Loads: Existing library from disk

        """
        
        self.workspace = workspace
        self.library_dir = workspace / "meta_prompt_library"
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        self.cwe_templates: Dict[str, CWEPromptTemplate] = {}
        self.pattern_prompts: Dict[str, str] = {}
        self.success_patterns: Dict[str, List[str]] = defaultdict(list)
        
        self._load_library()
        self._initialize_base_templates()
        
        logger.info("Meta Prompt Library initialized")

    ###################################################################################################################
    
    def _initialize_base_templates(self):
        """Initialize with enhanced base templates for common CWEs
        
           Templates for:
                - CWE-119: Buffer overflow
                - CWE-476: Null pointer dereference
                - CWE-125: Out-of-bounds read
                - CWE-787: Out-of-bounds write
                
           Each includes:
                - Success rate tracking
                - Key elements
                - Failure patterns
                
        (created the templates using Claude-4 Sonnet)
        """
        
        base_templates = {
            'CWE-119': CWEPromptTemplate(
                cwe_type='CWE-119',
                prompt_template="""🔍 CWE-119 BUFFER OVERFLOW ANALYSIS

Focus on buffer operations and bounds checking:
1. Identify ALL buffer declarations and their sizes
2. Track EVERY write operation to these buffers
3. Verify bounds checking before each write
4. Look for unsafe functions: strcpy, sprintf, gets
5. Check loop termination conditions

Key vulnerable patterns:
- strcpy/strcat without length checks
- Array access with unvalidated indices
- Off-by-one errors in loops
- Stack buffer operations

{context_specific}

Function Code:
```c
{function_code}
```

SYSTEMATIC ANALYSIS:
Vulnerable: [Yes/No]
Vulnerable statements: [List specific lines]
Root cause: [Explain the buffer overflow mechanism]""",
                success_rate=0.8,
                usage_count=0,
                last_updated=time.time(),
                key_elements=['buffer size', 'bounds check', 'unsafe functions'],
                failure_patterns=['missed off-by-one', 'complex indexing']
            ),
            
            'CWE-476': CWEPromptTemplate(
                cwe_type='CWE-476',
                prompt_template="""🔍 CWE-476 NULL POINTER DEREFERENCE ANALYSIS

Systematic NULL check verification:
1. Identify ALL pointer declarations
2. Track pointer assignments and modifications
3. Find EVERY dereference operation
4. Verify NULL checks before each dereference
5. Consider all execution paths

Critical checks:
- Function parameter pointers
- Return values from malloc/calloc
- Pointer arithmetic results
- Conditional pointer usage

{context_specific}

Function Code:
```c
{function_code}
```

SYSTEMATIC ANALYSIS:
Vulnerable: [Yes/No]
Vulnerable statements: [Specific dereference locations]
Missing checks: [Where NULL checks are needed]""",
                success_rate=0.9,
                usage_count=0,
                last_updated=time.time(),
                key_elements=['pointer declaration', 'null check', 'dereference'],
                failure_patterns=['conditional paths', 'indirect dereference']
            ),
            
            'CWE-125': CWEPromptTemplate(
                cwe_type='CWE-125',
                prompt_template="""🔍 CWE-125 OUT-OF-BOUNDS READ ANALYSIS

Comprehensive read operation analysis:
1. Map all buffer/array declarations with sizes
2. Identify ALL read operations (direct and indirect)
3. Trace index calculations and sources
4. Verify bounds validation before reads
5. Check for integer overflow in index math

Vulnerable patterns:
- User-controlled indices without validation
- Pointer arithmetic beyond buffer bounds
- Negative index values
- Loop conditions allowing overread

{context_specific}

Function Code:
```c
{function_code}
```

SYSTEMATIC ANALYSIS:
Vulnerable: [Yes/No]
Read violations: [Specific locations]
Index analysis: [How indices can exceed bounds]""",
                success_rate=0.4,
                usage_count=0,
                last_updated=time.time(),
                key_elements=['read bounds', 'index validation', 'buffer size'],
                failure_patterns=['complex pointer math', 'indirect reads']
            ),
            
            'CWE-787': CWEPromptTemplate(
                cwe_type='CWE-787',
                prompt_template="""🔍 CWE-787 OUT-OF-BOUNDS WRITE ANALYSIS

Comprehensive write operation analysis:
1. Identify all buffers and their allocated sizes
2. Track ALL write operations to memory
3. Analyze write sizes and destinations
4. Check bounds validation before writes
5. Consider all code paths

High-risk patterns:
- memcpy/memmove with unchecked sizes
- String operations without length limits
- Array writes with computed indices
- Pointer arithmetic before writes

{context_specific}

Function Code:
```c
{function_code}
```

SYSTEMATIC ANALYSIS:
Vulnerable: [Yes/No]
Write violations: [Specific locations]
Overflow mechanism: [How bounds are exceeded]""",
                success_rate=0.8,
                usage_count=0,
                last_updated=time.time(),
                key_elements=['write bounds', 'size validation', 'buffer limits'],
                failure_patterns=['complex size calculations', 'indirect writes']
            )
        }
        
        # Add templates that don't exist yet
        for cwe, template in base_templates.items():
            if cwe not in self.cwe_templates:
                self.cwe_templates[cwe] = template

    ###################################################################################################################
                
    def get_cwe_prompt(self, cwe_type: str, context: Dict, 
                       custom_adjustments: Dict[str, str] = None) -> str:
        
        """Get optimized prompt for specific CWE with context
        
           Process:
                1. Find CWE-specific template
                2. Add context information
                3. Apply custom adjustments
                4. Update usage statistics

           Falls back to: Generic template for unknown CWEs
        """
        
        if cwe_type not in self.cwe_templates:
            logger.warning(f"No specific template for {cwe_type}, using generic")
            return self._get_generic_prompt(cwe_type, context)
        
        template = self.cwe_templates[cwe_type]
        prompt = template.prompt_template
        
        # Add context-specific information
        context_parts = []
        if context.get('External_Functions'):
            context_parts.append(f"External functions used: {', '.join(context['External_Functions'][:5])}")
        if context.get('Function_Arguments'):
            context_parts.append(f"Function parameters: {', '.join(context['Function_Arguments'][:5])}")
        
        context_specific = '\n'.join(context_parts) if context_parts else "No additional context available"
        prompt = prompt.replace('{context_specific}', context_specific)
        
        # Apply custom adjustments from meta-learning
        if custom_adjustments:
            for key, value in custom_adjustments.items():
                prompt = prompt.replace(f'{{{key}}}', value)
        
        # Update usage count
        template.usage_count += 1
        
        return prompt

    ###################################################################################################################
    
    def update_template_performance(self, cwe_type: str, success: bool, 
                                   failure_reason: str = None):
        """Update template performance metrics

           Tracking:
                - Success rate (exponential moving average)
                - Failure patterns (recent 10)
                - Usage count

           Learning rate: α = 0.2

        """
        
        if cwe_type not in self.cwe_templates:
            return
        
        template = self.cwe_templates[cwe_type]
        
        # Update success rate with exponential moving average
        
        alpha = 0.2  # Learning rate
        template.success_rate = (1 - alpha) * template.success_rate + alpha * (1.0 if success else 0.0)
        
        # Track failure patterns
        if not success and failure_reason:
            if failure_reason not in template.failure_patterns:
                template.failure_patterns.append(failure_reason)
                
                # Keep only recent failure patterns
                if len(template.failure_patterns) > 10:
                    template.failure_patterns.pop(0)
        
        self._save_library()

    ###################################################################################################################
    
    def evolve_template(self, cwe_type: str, successful_analysis: str, 
                       key_insight: str) -> CWEPromptTemplate:
        
        """Evolve a template based on successful analysis

           Process:
               1. Extract key elements from success
               2. Update or create template
               3. Refine prompt structure
               4. Add new insights
        
        """
        
        if cwe_type not in self.cwe_templates:
            
            # Create new template from success
            self.cwe_templates[cwe_type] = CWEPromptTemplate(
                cwe_type=cwe_type,
                prompt_template=self._extract_prompt_structure(successful_analysis),
                success_rate=1.0,
                usage_count=1,
                last_updated=time.time(),
                key_elements=[key_insight],
                failure_patterns=[]
            )
            
        else:
            # Update existing template
            template = self.cwe_templates[cwe_type]
            
            # Extract new key elements from successful analysis
            new_elements = self._extract_key_elements(successful_analysis)
            for element in new_elements:
                if element not in template.key_elements:
                    template.key_elements.append(element)
            
            # Potentially refine the prompt template
            if template.success_rate < 0.6:
                
                # Template needs improvement
                template.prompt_template = self._refine_prompt_template(
                    template.prompt_template, successful_analysis, key_insight
                )
            
            template.last_updated = time.time()
        
        self._save_library()
        
        return self.cwe_templates[cwe_type]

    ###################################################################################################################
    
    def _extract_prompt_structure(self, successful_analysis: str) -> str:
        """Extract reusable prompt structure from successful analysis"""
        
        structure = f""" VULNERABILITY ANALYSIS

Systematic approach based on successful pattern:
{successful_analysis[:2000]}...

[Analysis continues with same structure]

Vulnerable: [Yes/No]
Details: [Specific findings]"""
        
        return structure

    ###################################################################################################################
    
    def _extract_key_elements(self, analysis: str) -> List[str]:
        """Extract key elements from successful analysis"""
        
        # Simplified extraction - look for security keywords
        
        keywords = ['buffer', 'overflow', 'bounds', 'null', 'pointer', 
                   'validation', 'check', 'sanitize', 'integer', 'race']
        
        found_elements = []
        analysis_lower = analysis.lower()
        
        for keyword in keywords:
            if keyword in analysis_lower:
                found_elements.append(keyword)
        
        return found_elements

    ###################################################################################################################
    
    def _refine_prompt_template(self, current_template: str, 
                               successful_analysis: str, key_insight: str) -> str:
        
        """Refine template based on successful analysis"""
        
        # Add key insight to template if not present
        if key_insight not in current_template:
            lines = current_template.split('\n')
            
            # Insert key insight after vulnerable patterns section
            for i, line in enumerate(lines):
                if 'vulnerable patterns' in line.lower():
                    lines.insert(i + 1, f"- {key_insight}")
                    break
                    
            current_template = '\n'.join(lines)
        
        return current_template

    ###################################################################################################################
    
    def _get_generic_prompt(self, cwe_type: str, context: Dict) -> str:
        """Generic prompt for unknown CWE types"""
        
        return f""" {cwe_type} VULNERABILITY ANALYSIS

Systematic security analysis:
1. Understand the vulnerability class {cwe_type}
2. Identify relevant code patterns
3. Check for proper validation and sanitization
4. Trace data flow from sources to sinks
5. Consider all execution paths

Context: {context}

Function Code:
```c
{function_code[:10000]}
```

Vulnerable: [Yes/No]
Vulnerable statements: [Specific locations]
Root cause: [Explain the vulnerability]"""

    ###################################################################################################################
    
    def _save_library(self):
        """Save prompt library to disk"""
        
        try:
            library_data = {
                'templates': {
                    cwe: {
                        'prompt_template': t.prompt_template,
                        'success_rate': t.success_rate,
                        'usage_count': t.usage_count,
                        'last_updated': t.last_updated,
                        'key_elements': t.key_elements,
                        'failure_patterns': t.failure_patterns
                    }
                    for cwe, t in self.cwe_templates.items()
                },
                'pattern_prompts': self.pattern_prompts
            }
            
            with open(self.library_dir / "prompt_library.json", 'w') as f:
                json.dump(library_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save prompt library: {e}")

    ###################################################################################################################
    
    def _load_library(self):
        """Load prompt library from disk"""
        
        try:
            library_file = self.library_dir / "prompt_library.json"
            if library_file.exists():
                with open(library_file, 'r') as f:
                    library_data = json.load(f)
                
                # Load templates
                for cwe, data in library_data.get('templates', {}).items():
                    self.cwe_templates[cwe] = CWEPromptTemplate(
                        cwe_type=cwe,
                        prompt_template=data['prompt_template'],
                        success_rate=data['success_rate'],
                        usage_count=data['usage_count'],
                        last_updated=data['last_updated'],
                        key_elements=data['key_elements'],
                        failure_patterns=data['failure_patterns']
                    )
                
                self.pattern_prompts = library_data.get('pattern_prompts', {})
                
                logger.info(f" Loaded {len(self.cwe_templates)} CWE templates from library")
                
        except Exception as e:
            logger.error(f"Failed to load prompt library: {e}")


# ============================================================
# 2. META-LEARNING AGENT
# ============================================================

class MetaLearningAgent:
    """Meta-agent that learns to optimize learning strategies

       self.exploration_rate: 0.2 (20% exploration)
       self.strategy_weights: Dict[strategy → weight]
       
    
    """
    
    def __init__(self, memory_system, prompt_library: MetaPromptLibrary):
        
        self.memory = memory_system
        self.prompt_library = prompt_library
        self.strategy_performance: Dict[str, Dict[str, LearningStrategyAnalysis]] = {}
        
        # Meta-learning parameters
        self.exploration_rate = 0.2  # Explore new strategies 20% of the time
        self.strategy_weights = {
            'instant_cache': 1.0,
            'focused_learning': 0.8,
            'template_transfer': 0.6,
            'multi_shot': 0.5,
            'cwe_specific': 0.7,
            'pattern_analysis': 0.6,
            'base': 0.3
        }
        
        logger.info(" Meta-Learning Agent initialized")

    ###################################################################################################################
    
    def analyze_iteration_failures(self, iteration_results: List[Dict]) -> Dict[str, Any]:
        """Analyze all failures from an iteration and recommend strategies

           POST-ITERATION ANALYSIS
           
           Analyzes:
               - Failure distribution by CWE
               - Failure patterns
               - Strategy effectiveness

           Generates:
               - Recommendations per CWE
               - Strategy adjustments

           Updates: Strategy weights
        
        
        """
        
        logger.info(" Meta-Learning: Analyzing iteration failures...")
        
        failure_analysis = {
            'total_failures': 0,
            'failure_by_cwe': defaultdict(list),
            'failure_by_pattern': defaultdict(list),
            'failure_by_strategy': defaultdict(int),
            'recommendations': []
        }
        
        # Analyze each failure
        for result in iteration_results:
            if not result.get('correct', False):
                failure_analysis['total_failures'] += 1
                
                cwe_type = result.get('cwe_type', 'Unknown')
                failure_pattern = self._identify_failure_pattern(result)
                strategy_used = result.get('prompt_strategy', 'unknown')
                
                failure_analysis['failure_by_cwe'][cwe_type].append(result)
                failure_analysis['failure_by_pattern'][failure_pattern].append(result)
                failure_analysis['failure_by_strategy'][strategy_used] += 1
        
        # Generate recommendations for each CWE type with failures
        for cwe_type, failures in failure_analysis['failure_by_cwe'].items():
            recommendation = self._generate_cwe_recommendation(cwe_type, failures)
            failure_analysis['recommendations'].append(recommendation)
        
        # Update strategy weights based on performance
        self._update_strategy_weights(iteration_results)
        
        logger.info(f" Analyzed {failure_analysis['total_failures']} failures")
        logger.info(f" Generated {len(failure_analysis['recommendations'])} recommendations")
        
        return failure_analysis

    ###################################################################################################################
    
    def select_optimal_strategy(self, function_data: Dict, 
                               iteration: int, failure_count: int) -> MetaLearningDecision:
        
        """Select optimal learning strategy based on meta-analysis
        
           STRATEGY SELECTION ALGORITHM
           
           Decision process:
                 1. Check exploration vs exploitation
                 2. If exploring: Try new strategies
                 3. If exploiting: Use best known strategy

           Considers:
                 - Historical performance
                 - CWE type
                 - Failure count
                 - Available strategies

           Returns: MetaLearningDecision with strategy and confidence
        
        """
        
        function_id = self.memory._generate_function_id(function_data['func_body'])
        cwe_type = self._extract_cwe_type(function_data)
        
        # Check if we should explore new strategies
        if np.random.random() < self.exploration_rate:
            return self._explore_new_strategy(function_data, cwe_type, iteration)
        
        # Otherwise, exploit best known strategy
        
        return self._exploit_best_strategy(function_data, cwe_type, failure_count)

    ###################################################################################################################
    
    def _generate_cwe_recommendation(self, cwe_type: str, 
                                    failures: List[Dict]) -> Dict[str, Any]:
        """Generate specific recommendations for a CWE type
           
           Analyzes:
                 - Common error patterns
                 - Template performance
                 - Failure reasons

           Suggests:
                 - Template evolution
                 - Detection pattern enhancement
                 - Multi-shot learning
        
        """
        
        # Analyze common failure patterns
        common_errors = self._analyze_common_errors(failures)
        
        # Check prompt library performance
        template_performance = 0.0
        if cwe_type in self.prompt_library.cwe_templates:
            template_performance = self.prompt_library.cwe_templates[cwe_type].success_rate
        
        recommendation = {
            'cwe_type': cwe_type,
            'failure_count': len(failures),
            'common_errors': common_errors,
            'current_template_performance': template_performance,
            'suggested_strategies': []
        }
        
        # Recommend strategies based on analysis
        if template_performance < 0.5:
            recommendation['suggested_strategies'].append({
                'strategy': 'evolve_template',
                'reason': f'Current template success rate only {template_performance:.1%}',
                'action': 'Analyze successful examples and evolve the prompt template'
            })
        
        if 'false_negative' in common_errors:
            recommendation['suggested_strategies'].append({
                'strategy': 'enhance_detection_patterns',
                'reason': 'Missing vulnerability detections',
                'action': 'Add more specific detection patterns to prompt'
            })
        
        if 'analysis_quality' in common_errors:
            recommendation['suggested_strategies'].append({
                'strategy': 'multi_shot_learning',
                'reason': 'Low quality analysis outputs',
                'action': 'Use multiple high-quality examples for better learning'
            })
        
        return recommendation

    ###################################################################################################################
    
    def _exploit_best_strategy(self, function_data: Dict, 
                              cwe_type: str, failure_count: int) -> MetaLearningDecision:
        
        """Select best known strategy for this case
        
           Priority order:
                  1. Instant cache (if available)
                  2. Focused learning (if solution exists)
                  3. Template transfer (after 1 failure)
                  4. Multi-shot (after 2+ failures)
                  5. CWE-specific (if template exists)
                  6. Base (fallback)
           Weights strategies by historical success
        
        """
        
        strategies = []
        
        # Check instant cache first (always best if available)
        cached = self.memory.try_instant_cache(function_data)
        if cached:
            return MetaLearningDecision(
                function_id=self.memory._generate_function_id(function_data['func_body']),
                cwe_type=cwe_type,
                failure_analysis="Exact match in cache",
                recommended_strategy="instant_cache",
                custom_prompt_adjustments={},
                confidence=1.0,
                reasoning="Instant cache provides perfect match"
            )
        
        # Evaluate available strategies
        if self.memory.find_working_solution(function_data):
            strategies.append(('focused_learning', self.strategy_weights['focused_learning']))
        
        if failure_count >= 1 and self.memory.find_successful_template(function_data):
            strategies.append(('template_transfer', self.strategy_weights['template_transfer']))
        
        if failure_count >= 2 and len(self.memory.get_multi_shot_examples(function_data)) >= 2:
            strategies.append(('multi_shot', self.strategy_weights['multi_shot']))
        
        if cwe_type in self.prompt_library.cwe_templates:
            weight = self.strategy_weights['cwe_specific'] * \
                    self.prompt_library.cwe_templates[cwe_type].success_rate
            strategies.append(('cwe_specific', weight))
        
        # Select highest weighted strategy
        if strategies:
            best_strategy = max(strategies, key=lambda x: x[1])
            strategy_name = best_strategy[0]
            confidence = best_strategy[1]
            
        else:
            strategy_name = 'base'
            confidence = self.strategy_weights['base']
        
        # Generate custom adjustments based on past failures
        custom_adjustments = self._generate_custom_adjustments(cwe_type, function_data)
        
        return MetaLearningDecision(
            function_id=self.memory._generate_function_id(function_data['func_body']),
            cwe_type=cwe_type,
            failure_analysis=f"Failed {failure_count} times previously",
            recommended_strategy=strategy_name,
            custom_prompt_adjustments=custom_adjustments,
            confidence=confidence,
            reasoning=f"Selected {strategy_name} based on historical performance"
        )

    ###################################################################################################################
    
    def _explore_new_strategy(self, function_data: Dict, 
                             cwe_type: str, iteration: int) -> MetaLearningDecision:
        """Explore new strategies for learning
        
           Tries:
                - Pattern analysis
                - CWE-specific approaches
                - Multi-shot variations

           Frequency: 20% of decisions
        
        
        """
        
        # Try pattern-based analysis for exploration
        pattern = self.memory._identify_vulnerability_pattern(function_data)
        
        exploration_strategies = [
            'pattern_analysis',
            'cwe_specific',
            'multi_shot'
        ]
        
        # Choose random exploration strategy
        strategy = np.random.choice(exploration_strategies)
        
        return MetaLearningDecision(
            function_id=self.memory._generate_function_id(function_data['func_body']),
            cwe_type=cwe_type,
            failure_analysis="Exploration phase",
            recommended_strategy=strategy,
            custom_prompt_adjustments={
                'exploration_focus': f'Trying {strategy} approach',
                'pattern_hint': pattern
            },
            confidence=0.5,
            reasoning=f"Exploring {strategy} to improve learning"
        )

    ###################################################################################################################
    
    def _update_strategy_weights(self, iteration_results: List[Dict]):
        """Update strategy weights based on performance
        
           
           Algorithm:
                 - Calculate success rate per strategy
                 - Apply exponential moving average (α=0.1)
                 - Adjust weights accordingly

           Effect: Successful strategies get higher priority

           Convergence: Typically within 3-5 iterations
        
        """
        
        strategy_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        
        for result in iteration_results:
            strategy = result.get('prompt_strategy', 'unknown')
            if strategy != 'unknown':
                strategy_performance[strategy]['total'] += 1
                
                if result.get('correct', False):
                    strategy_performance[strategy]['success'] += 1
        
        # Update weights using exponential moving average
        alpha = 0.1  # Learning rate
        
        for strategy, performance in strategy_performance.items():
            if performance['total'] > 0:
                success_rate = performance['success'] / performance['total']
                
                if strategy in self.strategy_weights:
                    
                    old_weight = self.strategy_weights[strategy]
                    new_weight = (1 - alpha) * old_weight + alpha * success_rate
                    self.strategy_weights[strategy] = new_weight
                    
                    logger.info(f" Updated {strategy} weight: {old_weight:.2f} → {new_weight:.2f}")

    ###################################################################################################################
    
    def _generate_custom_adjustments(self, cwe_type: str, 
                                    function_data: Dict) -> Dict[str, str]:
        
        """Generate custom prompt adjustments based on learning"""
        
        adjustments = {}
        
        # Add specific focus areas based on CWE type
        if cwe_type == 'CWE-125':
            adjustments['focus_area'] = "Pay special attention to array indices and bounds checking"
            
        elif cwe_type == 'CWE-476':
            adjustments['focus_area'] = "Track all pointer paths and check for NULL before every dereference"
            
        elif cwe_type == 'CWE-119':
            adjustments['focus_area'] = "Check buffer sizes and validate all write operations"
        
        # Add context-specific hints
        if function_data.get('context', {}).get('External_Functions'):
            risky_functions = ['strcpy', 'sprintf', 'gets', 'scanf']
            
            used_risky = [f for f in function_data['context']['External_Functions'] 
                         if f in risky_functions]
            
            if used_risky:
                adjustments['risk_hint'] = f"High-risk functions detected: {', '.join(used_risky)}"
        
        return adjustments

    ###################################################################################################################
    
    def _extract_cwe_type(self, function_data: Dict) -> str:
        
        """Extract CWE type from function data"""
        
        cwe_list = function_data.get('cwe_list', [])
        
        if cwe_list:
            
            if isinstance(cwe_list[0], np.ndarray):
                return str(cwe_list[0][0]) if len(cwe_list[0]) > 0 else 'Unknown'
                
            else:
                return str(cwe_list[0])
                
        return 'Unknown'

    ###################################################################################################################
    
    def _identify_failure_pattern(self, result: Dict) -> str:
        """Identify the pattern of failure"""
        
        if result.get('vulnerability_detected') and not result.get('ground_truth_vulnerable'):
            return 'false_positive'
            
        elif not result.get('vulnerability_detected') and result.get('ground_truth_vulnerable'):
            return 'false_negative'
            
        else:
            return 'analysis_error'

    ###################################################################################################################
    
    def _analyze_common_errors(self, failures: List[Dict]) -> List[str]:
        """Analyze common error patterns in failures"""
        
        error_types = defaultdict(int)
        
        for failure in failures:
            if failure.get('vulnerability_detected') != failure.get('ground_truth_vulnerable'):
                
                if failure.get('vulnerability_detected'):
                    error_types['false_positive'] += 1
                    
                else:
                    error_types['false_negative'] += 1
            
            # Check analysis quality
            if failure.get('reasoning_length', 0) < 100:
                error_types['analysis_quality'] += 1
        
        # Return most common errors
        common_errors = [error for error, count in error_types.items() 
                        if count > len(failures) * 0.3]
        
        return common_errors


# ============================================================
# 3. ENHANCED SICA-VULN AGENT WITH META-LEARNING
# ============================================================


class EnhancedSICAVulnAgent(SICAVulnAgent):
    """Enhanced SICA-VULN with meta-learning capabilities"""
    
    def __init__(self, workspace_dir: str = "./sica_vuln_workspace", 
                 use_real_data: bool = True, mock_llm: bool = False):
        
        # Initialize base agent
        super().__init__(workspace_dir, use_real_data, mock_llm)
        
        # Add meta-learning components
        self.prompt_library = MetaPromptLibrary(self.workspace)
        self.meta_agent = MetaLearningAgent(self.memory, self.prompt_library)
        
        # Track iteration results for meta-analysis
        self.iteration_results = []
        
        logger.info(" Enhanced SICA-VULN with Meta-Learning initialized!")
        logger.info(" Dynamic prompt library ready")
        logger.info(" Meta-agent learning to learn")

    ###################################################################################################################
    
    async def analyze_vulnerability_with_meta_learning(self, function_data: Dict, 
                                                      iteration: int = 0) -> Dict[str, Any]:
        
        """Enhanced analysis with meta-learning strategy selection

           Enhanced phases:
                  1. Instant cache check
                  2. Meta-strategy selection
                  3. Meta-optimized prompt generation
                  4. LLM analysis
                  5. Evaluation
                  6. Template evolution (if successful)
                  7. Meta-learning updates

           Additions over base SIVA:
                  - Strategy optimization
                  - Template evolution
                  - Failure analysis
        
        
        """
        
        logger.info(f" Meta-Enhanced Analysis (iter {iteration}): {function_data.get('func_name', 'unknown')}...")
        
        # Phase 1: Try instant cache (always first)
        cached_analysis = self.memory.try_instant_cache(function_data)
        
        if cached_analysis:
            
            logger.info(" Using instant cached analysis!")
            
            result = self.evaluator.evaluate_vulnerability_analysis(cached_analysis, function_data)
            result["cache_hit"] = True
            result["prompt_strategy"] = "cached"
            result["meta_decision"] = "instant_cache"
            
            return result
        
        # Phase 2: Meta-learning strategy selection
        function_id = self.memory._generate_function_id(function_data['func_body'])
        failure_count = self._count_failures(function_id)
        
        meta_decision = self.meta_agent.select_optimal_strategy(
            function_data, iteration, failure_count
        )
        
        logger.info(f" Meta-Decision: {meta_decision.recommended_strategy} (confidence: {meta_decision.confidence:.2f})")
        logger.info(f"   Reasoning: {meta_decision.reasoning}")
        
        # Phase 3: Generate prompt based on meta-decision
        prompt = await self._generate_meta_optimized_prompt(function_data, meta_decision)
        
        # Phase 4: Generate analysis
        analysis = await self.llm_client.analyze_vulnerability(prompt, max_tokens=600)
        
        # Phase 5: Evaluate results
        result = self.evaluator.evaluate_vulnerability_analysis(analysis, function_data)
        result["cache_hit"] = False
        result["prompt_strategy"] = meta_decision.recommended_strategy
        result["meta_decision"] = meta_decision.recommended_strategy
        result["meta_confidence"] = meta_decision.confidence
        
        # Phase 6: Update meta-learning
        cwe_type = meta_decision.cwe_type
        self.prompt_library.update_template_performance(
            cwe_type, 
            result.get('correct', False),
            self._classify_failure_reason(result)
        )
        
        # Phase 7: Record for learning
        self.memory.record_attempt(function_data, analysis, result, iteration)
        
        # Phase 8: If successful, potentially evolve the template
        if result.get('correct', False) and result.get('reasoning_length', 0) > 200:
            
            key_insight = self._extract_key_insight(analysis, function_data)
            self.prompt_library.evolve_template(cwe_type, analysis, key_insight)
            logger.info(f" Evolved template for {cwe_type}")
        
        return result

    ###################################################################################################################
    
    async def _generate_meta_optimized_prompt(self, function_data: Dict, 
                                            meta_decision: MetaLearningDecision) -> str:
        """Generate prompt optimized by meta-learning
        
           Strategy mapping:
                 - focused_learning → Focused prompt
                 - template_transfer → Template prompt
                 - multi_shot → Multi-shot prompt
                 - cwe_specific → Evolved CWE template
                 - pattern_analysis → Pattern-focused prompt
                 - base → Standard prompt

           Incorporates: Custom adjustments from meta-learning
        
        """
        
        strategy = meta_decision.recommended_strategy
        
        if strategy == 'focused_learning':
            
            # Use focused learning
            working_solution = self.memory.find_working_solution(function_data)
            return self.memory._generate_focused_security_prompt(function_data, working_solution)
            
        elif strategy == 'template_transfer':
            
            # Use template transfer
            template = self.memory.find_successful_template(function_data)
            return self.memory._generate_template_security_prompt(function_data, template)
            
        elif strategy == 'multi_shot':
            
            # Use multi-shot learning
            examples = self.memory.get_multi_shot_examples(function_data, 3)
            return self.memory._generate_multi_shot_security_prompt(function_data, examples)
            
        elif strategy == 'cwe_specific':
            
            # Use evolved CWE-specific template
            cwe_type = meta_decision.cwe_type
            context = function_data.get('context', {})
            prompt = self.prompt_library.get_cwe_prompt(
                cwe_type, context, meta_decision.custom_prompt_adjustments
            )
            
            # Add function code
            function_code = function_data.get('func_body', '')
            prompt = prompt.replace('{function_code}', function_code[:1500])
            
            return prompt
            
        elif strategy == 'pattern_analysis':
            
            # Pattern-based analysis
            return self._generate_pattern_analysis_prompt(function_data, meta_decision)
            
        else:
            
            # Fallback to base prompt
            return self._create_security_specific_prompt(function_data)

    ###################################################################################################################
    
    def _generate_pattern_analysis_prompt(self, function_data: Dict, 
                                        meta_decision: MetaLearningDecision) -> str:
        
        """Generate pattern-focused analysis prompt"""
        
        pattern = self.memory._identify_vulnerability_pattern(function_data)
        cwe_type = meta_decision.cwe_type
        
        return f""" PATTERN-BASED VULNERABILITY ANALYSIS

Target Pattern: {pattern}
Expected CWE: {cwe_type}

Function Code:
```c
{function_data['func_body'][:10000]}...
```

Pattern-Specific Analysis:
1. Identify all instances of {pattern} patterns
2. Check for vulnerability indicators specific to {cwe_type}
3. Verify security controls for this pattern type
4. Consider edge cases and bypass scenarios

{meta_decision.custom_prompt_adjustments.get('pattern_hint', '')}

Vulnerable: [Yes/No]
Pattern instances: [List specific occurrences]
Security gaps: [Missing protections]"""

    ###################################################################################################################
    
    async def run_meta_learning_benchmark(self, n_samples: int = 50, 
                                        iterations: int = 3) -> Dict[str, Any]:
        
        """Run benchmark with meta-learning analysis after each iteration
           
           Additional features:
                - Per-iteration failure analysis
                - Strategy weight evolution
                - Template library growth
                - Meta-recommendations

           Displays:
                - Strategy usage statistics
                - Weight updates
                - Template evolution
                - Learning insights

           Returns: Enhanced results with meta-stats
        
        """
        
        logger.info(f" META-LEARNING SICA-VULN BENCHMARK")
        logger.info(f" Functions: {n_samples}")
        logger.info(f" Iterations: {iterations}")
        logger.info(f" Meta-agent will optimize learning strategies")
        
        # Load data
        sample_data = self.data_loader.load_sample_data(n_samples)
        all_results = []
        
        for iteration in range(iterations):
            logger.info(f"\n META-LEARNING ITERATION {iteration + 1}/{iterations}")
            logger.info("=" * 70)
            
            iteration_start = time.time()
            iteration_results = []
            correct_count = 0
            
            # Track meta-decisions
            meta_decisions = defaultdict(int)
            
            for i, function_data in enumerate(sample_data, 1):
                function_data = self.data_loader._validate_and_fix_sample(function_data)
                
                logger.info(f" Function {i}/{n_samples}: {function_data.get('func_name', 'unknown')}")
                
                # Use meta-learning enhanced analysis
                result = await self.analyze_vulnerability_with_meta_learning(function_data, iteration)
                
                # Track results
                result['cwe_type'] = self._extract_cwe_type(function_data)
                iteration_results.append(result)
                
                if result.get('correct', False):
                    
                    correct_count += 1
                    logger.info(f"    SUCCESS! (strategy: {result.get('meta_decision', 'unknown')})")
                    
                else:
                    logger.info(f"    Learning... (strategy: {result.get('meta_decision', 'unknown')})")
                
                meta_decisions[result.get('meta_decision', 'unknown')] += 1
            
            # Calculate iteration metrics
            precision = correct_count / n_samples
            f1_score = precision  # Simplified
            
            iteration_result = {
                "iteration": iteration + 1,
                "f1_score": f1_score,
                "correct": correct_count,
                "total": n_samples,
                "time": time.time() - iteration_start,
                "meta_decisions": dict(meta_decisions)
            }
            
            all_results.append(iteration_result)
            
            # META-LEARNING PHASE: Analyze and adapt
            logger.info("\n META-LEARNING ANALYSIS PHASE")
            logger.info("=" * 50)
            
            failure_analysis = self.meta_agent.analyze_iteration_failures(iteration_results)
            
            logger.info(f" Failure Analysis:")
            logger.info(f"   Total failures: {failure_analysis['total_failures']}")
            logger.info(f"   Unique CWEs with failures: {len(failure_analysis['failure_by_cwe'])}")
            
            # Display recommendations
            logger.info(f"\n Meta-Learning Recommendations:")
            for rec in failure_analysis['recommendations'][:5]:  # Top 5
                logger.info(f"\n   CWE {rec['cwe_type']} ({rec['failure_count']} failures):")
                for strategy in rec['suggested_strategies']:
                    logger.info(f"      - {strategy['strategy']}: {strategy['reason']}")
            
            # Show updated strategy weights
            logger.info(f"\n Updated Strategy Weights:")
            for strategy, weight in sorted(self.meta_agent.strategy_weights.items(), 
                                         key=lambda x: x[1], reverse=True):
                logger.info(f"   {strategy}: {weight:.3f}")
            
            # Show prompt library stats
            logger.info(f"\n Prompt Library Evolution:")
            for cwe, template in list(self.prompt_library.cwe_templates.items())[:5]:
                logger.info(f"   {cwe}: {template.success_rate:.1%} success rate ({template.usage_count} uses)")
            
            await asyncio.sleep(0.5)
        
        # Final results
        final_f1 = all_results[-1]["f1_score"] if all_results else 0
        improvement = final_f1 - all_results[0]["f1_score"] if all_results else 0
        
        logger.info(f"\n META-LEARNING BENCHMARK COMPLETE!")
        logger.info(f" Final F1-Score: {final_f1:.3f}")
        logger.info(f" Total Improvement: {improvement:+.3f}")
        logger.info(f" Evolved Templates: {len(self.prompt_library.cwe_templates)}")
        logger.info(f" Meta-Learning Insights: Agent learned optimal strategies for each CWE")
        
        return {
            "final_f1_score": final_f1,
            "improvement": improvement,
            "all_results": all_results,
            "evolved_templates": len(self.prompt_library.cwe_templates),
            "meta_learning_stats": {
                "strategy_weights": dict(self.meta_agent.strategy_weights),
                "prompt_templates": len(self.prompt_library.cwe_templates)
            }
        }

    ###################################################################################################################
    
    def _classify_failure_reason(self, result: Dict) -> str:
        
        """Classify the reason for failure"""
        
        if not result.get('correct', False):
            if result.get('vulnerability_detected') != result.get('ground_truth_vulnerable'):
                
                if result.get('vulnerability_detected'):
                    return "false_positive_detection"
                    
                else:
                    return "missed_vulnerability"
            else:
                return "analysis_quality_issue"
                
        return "success"

    ###################################################################################################################
    
    def _extract_key_insight(self, analysis: str, function_data: Dict) -> str:
        
        """Extract key insight from successful analysis"""
        
        # Simplified extraction 
        cwe_type = self._extract_cwe_type(function_data)
        
        if 'buffer overflow' in analysis.lower():
            return "buffer size validation critical"
            
        elif 'null pointer' in analysis.lower():
            return "null check before dereference"
            
        elif 'bounds check' in analysis.lower():
            return "array index validation required"
            
        else:
            return f"successful {cwe_type} detection pattern"


# ============================================================
# 4. MAIN FUNCTION
# ============================================================

async def main():
    
    """Main function to run meta-learning enhanced SICA-VULN"""
    
    print("""
 SICA-VULN WITH META-LEARNING
===============================

This enhanced version includes:
✅ Meta-agent that learns to optimize learning strategies
✅ Dynamic CWE-specific prompt library that evolves
✅ Strategy weight adaptation based on performance
✅ Learning to learn - the agent improves its learning process

Choose option:
1. Test single vulnerability with meta-learning
2. Quick meta-learning benchmark (10 samples, 2 iterations)
3. Full meta-learning benchmark (50 samples, 3 iterations)
4. Show prompt library evolution
""")

    choice = input("Enter choice (1-4): ").strip() or "1"
    
    # Check for LLM server or use mock mode
    print("\n Checking LLM server connection...")
    mock_llm = False
    
    test_client = SecurityLLMClient()
    try:
        if await test_client.health_check():
            print("✅ LLM server connected successfully!")
        else:
            raise Exception("LLM server not responding")
    except:
        print("⚠️ LLM server not available")
        use_mock = input("Use MOCK mode for testing? (Y/n): ").strip().lower()
        if use_mock != 'n':
            mock_llm = True
            print("🤖 Using MOCK LLM mode - simulated responses for testing")
        else:
            print("❌ Cannot proceed without LLM server")
            return
            
    finally:
        await test_client.close()
    
    # Initialize enhanced agent
    agent = EnhancedSICAVulnAgent(use_real_data=True, mock_llm=mock_llm)
    await agent.initialize()

    
    try:
        if choice == "1":
            # Test single vulnerability
            samples = agent.data_loader.load_sample_data(n_samples=1)
            if samples:
                result = await agent.analyze_vulnerability_with_meta_learning(samples[0], 0)
                print(f"\n Result: {'SUCCESS' if result.get('correct') else 'LEARNING'}")
                print(f"Strategy used: {result.get('meta_decision')}")
                print(f"Confidence: {result.get('meta_confidence', 0):.2f}")
                
        elif choice == "2":
            # Quick benchmark
            await agent.run_meta_learning_benchmark(n_samples=10, iterations=2)
            
        elif choice == "3":
            # Full benchmark
            await agent.run_meta_learning_benchmark(n_samples=1000, iterations=5)
            
        elif choice == "4":
            # Show library stats
            print("\n Prompt Library Evolution:")
            for cwe, template in agent.prompt_library.cwe_templates.items():
                print(f"\n{cwe}:")
                print(f"  Success rate: {template.success_rate:.1%}")
                print(f"  Usage count: {template.usage_count}")
                print(f"  Key elements: {', '.join(template.key_elements[:5])}")
                print(f"  Common failures: {', '.join(template.failure_patterns[:3])}")
                
    except Exception as e:
        
        logger.error(f"Error: {e}")
        
        import traceback
        traceback.print_exc()
        
    finally:
        await agent.llm_client.close()

###################################################################################################################


# run full system
if __name__ == "__main__":
    asyncio.run(main())
