"""
Code Reviewer Worker-Agent
Reviews generated code for syntax errors, logic bugs, and security issues before execution
"""

from typing import Dict, Any, List
from core.agents.base_agent import BaseWorker
from core.agents.prompts import (
    CODE_REVIEW_PROMPT,
    format_prompt
)
from shared.utils.data_processing import extract_and_validate_json
import ast
import re

class CodeReviewerWorker(BaseWorker):
    def __init__(self):
        super().__init__('CODE_REVIEWER')
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Reviews generated code before execution"""
        self.log('info', 'Starting code review')
        
        results = pipeline_state.get('results', {})
        
        # Get generated code
        code_gen_result = results.get('CODE_GENERATOR', {})
        if not code_gen_result:
            self.log('warning', 'No code generation result available - skipping review')
            return {
                'success': True,
                'reviewPerformed': False,
                'reason': 'No code to review'
            }
        
        # Extract code - handle both dict and string results
        if isinstance(code_gen_result, str):
            generated_code = code_gen_result
        elif isinstance(code_gen_result, dict):
            generated_code = code_gen_result.get('code', '')
        else:
            self.log('warning', f'Unexpected code generation result type: {type(code_gen_result)}')
            return {
                'success': True,
                'reviewPerformed': False,
                'reason': 'Invalid code format'
            }
        
        if not generated_code:
            self.log('warning', 'No code found in generation result')
            return {
                'success': True,
                'reviewPerformed': False,
                'reason': 'Empty code'
            }
        
        # Perform multi-level review
        review_result = await self._perform_comprehensive_review(
            code=generated_code,
            pipeline_state=pipeline_state
        )
        
        # Determine if code is safe to execute
        is_safe = self._assess_code_safety(review_result)
        
        result = {
            'success': True,
            'reviewPerformed': True,
            'isSafe': is_safe,
            'reviewResult': review_result,
            'originalCode': generated_code,
            'improvedCode': review_result.get('improvedCode', generated_code),
            'issues': review_result.get('issues', []),
            'recommendations': review_result.get('recommendations', []),
            'securityScore': review_result.get('securityScore', 0),
            'qualityScore': review_result.get('qualityScore', 0)
        }
        
        if is_safe:
            self.log('success', f'Code review passed - Quality: {result["qualityScore"]}/10, Security: {result["securityScore"]}/10')
        else:
            self.log('warning', f'Code review found critical issues - {len(result["issues"])} issues detected')
        
        return result
    
    async def _perform_comprehensive_review(
        self,
        code: str,
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Performs comprehensive code review using static analysis + LLM"""
        
        # Step 1: Static Analysis (Fast, deterministic)
        static_analysis = self._static_code_analysis(code)
        
        # Step 2: LLM-based Review (Deep, contextual)
        llm_review = await self._llm_code_review(code, pipeline_state, static_analysis)
        
        # Step 3: Merge results
        return self._merge_review_results(static_analysis, llm_review)
    
    def _static_code_analysis(self, code: str) -> Dict[str, Any]:
        """Performs static code analysis (syntax, basic security)"""
        issues = []
        warnings = []
        
        # 1. Syntax Check
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'severity': 'critical',
                'message': f'Syntax error at line {e.lineno}: {e.msg}',
                'line': e.lineno
            })
        
        # 2. Security Checks
        security_issues = self._check_security_patterns(code)
        issues.extend(security_issues)
        
        # 3. Code Quality Checks
        quality_issues = self._check_code_quality(code)
        warnings.extend(quality_issues)
        
        # 4. Best Practices
        best_practice_issues = self._check_best_practices(code)
        warnings.extend(best_practice_issues)
        
        return {
            'issues': issues,
            'warnings': warnings,
            'syntaxValid': len([i for i in issues if i['type'] == 'syntax_error']) == 0,
            'securityScore': self._calculate_security_score(security_issues),
            'qualityScore': self._calculate_quality_score(quality_issues)
        }
    
    def _check_security_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Checks for common security anti-patterns"""
        issues = []
        
        # Dangerous patterns
        dangerous_patterns = [
            (r'eval\s*\(', 'Use of eval() is dangerous'),
            (r'exec\s*\(', 'Use of exec() is dangerous'),
            (r'__import__\s*\(', 'Dynamic imports can be dangerous'),
            (r'os\.system\s*\(', 'Use of os.system() is dangerous'),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', 'Shell=True in subprocess is dangerous'),
            (r'pickle\.loads?\s*\(', 'Pickle can execute arbitrary code'),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                issues.append({
                    'type': 'security_risk',
                    'severity': 'critical',
                    'message': message,
                    'pattern': pattern
                })
        
        # Infinite loop detection (basic)
        if re.search(r'while\s+True\s*:', code) and 'break' not in code:
            issues.append({
                'type': 'infinite_loop',
                'severity': 'high',
                'message': 'Potential infinite loop detected (while True without break)'
            })
        
        return issues
    
    def _check_code_quality(self, code: str) -> List[Dict[str, Any]]:
        """Checks code quality issues"""
        issues = []
        
        # Check for hardcoded values
        if re.search(r'password\s*=\s*["\']', code, re.IGNORECASE):
            issues.append({
                'type': 'hardcoded_credential',
                'severity': 'medium',
                'message': 'Hardcoded password detected'
            })
        
        # Check for proper error handling
        if 'try:' in code and 'except:' in code:
            if re.search(r'except\s*:', code):
                issues.append({
                    'type': 'bare_except',
                    'severity': 'low',
                    'message': 'Bare except clause - should specify exception type'
                })
        
        # Check for print statements (should use logging)
        print_count = len(re.findall(r'\bprint\s*\(', code))
        if print_count > 5:
            issues.append({
                'type': 'excessive_prints',
                'severity': 'low',
                'message': f'Excessive print statements ({print_count}) - consider using logging'
            })
        
        return issues
    
    def _check_best_practices(self, code: str) -> List[Dict[str, Any]]:
        """Checks ML best practices"""
        issues = []
        
        # Check for train-test split
        if 'fit(' in code and 'train_test_split' not in code:
            issues.append({
                'type': 'missing_train_test_split',
                'severity': 'medium',
                'message': 'Model training without train-test split detected'
            })
        
        # Check for feature scaling (for certain algorithms)
        if any(algo in code for algo in ['SVC', 'SVR', 'KNeighbors', 'LogisticRegression']):
            if 'StandardScaler' not in code and 'MinMaxScaler' not in code:
                issues.append({
                    'type': 'missing_scaling',
                    'severity': 'medium',
                    'message': 'Algorithm may benefit from feature scaling'
                })
        
        return issues
    
    async def _llm_code_review(
        self,
        code: str,
        pipeline_state: Dict[str, Any],
        static_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Performs LLM-based code review"""
        
        project = pipeline_state.get('project', {})
        results = pipeline_state.get('results', {})
        
        # Build context
        context = self._build_review_context(project, results)
        
        # Create prompt
        hyperparam = results.get('HYPERPARAMETER_OPTIMIZER', {})
        algorithm = 'Unknown'
        if isinstance(hyperparam, dict):
            algorithm = hyperparam.get('algorithm', 'Unknown')
        
        prompt = format_prompt(CODE_REVIEW_PROMPT, {
            'code': code,
            'context': context,
            'staticAnalysis': self._format_static_analysis(static_analysis),
            'projectName': project.get('name', 'Unknown'),
            'algorithm': algorithm
        })
        
        try:
            # Call LLM
            self.log('info', 'Calling LLM for code review')
            response = await self.call_llm(prompt, None, self.config.get('maxTokens', 4096))
            
            # Extract JSON from response
            result = extract_and_validate_json(response)
            
            return {
                'issues': result.get('issues', []),
                'recommendations': result.get('recommendations', []),
                'improvedCode': result.get('improvedCode', code),
                'securityScore': result.get('securityScore', 5),
                'qualityScore': result.get('qualityScore', 5),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as error:
            self.log('error', f'LLM code review failed: {error}')
            
            # Fallback: Return static analysis only
            return {
                'issues': [],
                'recommendations': ['LLM review unavailable - relying on static analysis'],
                'improvedCode': code,
                'securityScore': static_analysis.get('securityScore', 5),
                'qualityScore': static_analysis.get('qualityScore', 5),
                'reasoning': 'Fallback: LLM review failed'
            }
    
    def _merge_review_results(
        self,
        static_analysis: Dict[str, Any],
        llm_review: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merges static analysis and LLM review results"""
        
        # Combine issues (prioritize static analysis for critical issues)
        all_issues = static_analysis.get('issues', []) + static_analysis.get('warnings', [])
        llm_issues = llm_review.get('issues', [])
        
        # Deduplicate issues
        combined_issues = all_issues + [
            issue for issue in llm_issues 
            if not self._is_duplicate_issue(issue, all_issues)
        ]
        
        # Calculate final scores (weighted average)
        static_security = static_analysis.get('securityScore', 5)
        llm_security = llm_review.get('securityScore', 5)
        final_security = (static_security * 0.6 + llm_security * 0.4)
        
        static_quality = static_analysis.get('qualityScore', 5)
        llm_quality = llm_review.get('qualityScore', 5)
        final_quality = (static_quality * 0.4 + llm_quality * 0.6)
        
        return {
            'issues': combined_issues,
            'recommendations': llm_review.get('recommendations', []),
            'improvedCode': llm_review.get('improvedCode', ''),
            'securityScore': round(final_security, 1),
            'qualityScore': round(final_quality, 1),
            'reasoning': llm_review.get('reasoning', ''),
            'syntaxValid': static_analysis.get('syntaxValid', False)
        }
    
    def _is_duplicate_issue(self, issue: Dict[str, Any], existing_issues: List[Dict[str, Any]]) -> bool:
        """Checks if an issue is a duplicate"""
        for existing in existing_issues:
            if existing.get('type') == issue.get('type'):
                return True
        return False
    
    def _assess_code_safety(self, review_result: Dict[str, Any]) -> bool:
        """Determines if code is safe to execute"""
        
        # Check syntax
        if not review_result.get('syntaxValid', False):
            return False
        
        # Check for critical issues
        issues = review_result.get('issues', [])
        critical_issues = [i for i in issues if i.get('severity') == 'critical']
        
        if critical_issues:
            return False
        
        # Check security score
        security_score = review_result.get('securityScore', 0)
        if security_score < 5:
            return False
        
        return True
    
    def _calculate_security_score(self, security_issues: List[Dict[str, Any]]) -> float:
        """Calculates security score (0-10)"""
        if not security_issues:
            return 10.0
        
        # Deduct points based on severity
        score = 10.0
        for issue in security_issues:
            severity = issue.get('severity', 'low')
            if severity == 'critical':
                score -= 3.0
            elif severity == 'high':
                score -= 2.0
            elif severity == 'medium':
                score -= 1.0
            else:
                score -= 0.5
        
        return max(0.0, score)
    
    def _calculate_quality_score(self, quality_issues: List[Dict[str, Any]]) -> float:
        """Calculates code quality score (0-10)"""
        if not quality_issues:
            return 10.0
        
        # Deduct points based on severity
        score = 10.0
        for issue in quality_issues:
            severity = issue.get('severity', 'low')
            if severity == 'high':
                score -= 2.0
            elif severity == 'medium':
                score -= 1.0
            else:
                score -= 0.5
        
        return max(0.0, score)
    
    def _build_review_context(self, project: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Builds context for LLM review"""
        lines = []
        
        lines.append(f"Project: {project.get('name', 'Unknown')}")
        lines.append(f"CSV File: {project.get('csvFilePath', 'Unknown')}")
        
        # Add data analysis context
        data_analysis = results.get('DATA_ANALYZER', {})
        if data_analysis and isinstance(data_analysis, dict):
            lines.append(f"\nData Analysis:")
            exploration = data_analysis.get('exploration', {})
            if isinstance(exploration, dict):
                lines.append(f"  - Rows: {exploration.get('rowCount', 'Unknown')}")
                columns = exploration.get('columns', [])
                if isinstance(columns, list):
                    lines.append(f"  - Columns: {len(columns)}")
        
        # Add hyperparameter context
        hyperparam = results.get('HYPERPARAMETER_OPTIMIZER', {})
        if hyperparam and isinstance(hyperparam, dict):
            lines.append(f"\nAlgorithm: {hyperparam.get('algorithm', 'Unknown')}")
            lines.append(f"Model Type: {hyperparam.get('modelType', 'Unknown')}")
        
        return "\n".join(lines)
    
    def _format_static_analysis(self, static_analysis: Dict[str, Any]) -> str:
        """Formats static analysis for prompt"""
        lines = []
        
        lines.append(f"Syntax Valid: {static_analysis.get('syntaxValid', False)}")
        lines.append(f"Security Score: {static_analysis.get('securityScore', 0)}/10")
        lines.append(f"Quality Score: {static_analysis.get('qualityScore', 0)}/10")
        
        issues = static_analysis.get('issues', [])
        if issues:
            lines.append(f"\nCritical Issues Found: {len(issues)}")
            for issue in issues[:5]:  # Limit to 5
                lines.append(f"  - {issue.get('message', 'Unknown issue')}")
        
        warnings = static_analysis.get('warnings', [])
        if warnings:
            lines.append(f"\nWarnings Found: {len(warnings)}")
            for warning in warnings[:5]:  # Limit to 5
                lines.append(f"  - {warning.get('message', 'Unknown warning')}")
        
        return "\n".join(lines)
