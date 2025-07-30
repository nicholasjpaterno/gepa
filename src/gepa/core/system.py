"""Compound AI System implementation."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from pydantic import BaseModel


@dataclass
class IOSchema:
    """Input/Output schema definition."""
    fields: Dict[str, type]
    required: List[str]
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against schema."""
        # Check required fields
        for field in self.required:
            if field not in data:
                return False
        
        # Check field types
        for field, value in data.items():
            if field in self.fields:
                expected_type = self.fields[field]
                if not isinstance(value, expected_type):
                    return False
        
        return True


@dataclass
class LanguageModule:
    """Language module component."""
    id: str
    prompt: str
    model_weights: Optional[str] = None  # Model identifier
    input_schema: Optional[IOSchema] = None
    output_schema: Optional[IOSchema] = None
    
    def update_prompt(self, new_prompt: str) -> "LanguageModule":
        """Create a new module with updated prompt."""
        return LanguageModule(
            id=self.id,
            prompt=new_prompt,
            model_weights=self.model_weights,
            input_schema=self.input_schema,
            output_schema=self.output_schema
        )


@runtime_checkable  
class ControlFlow(Protocol):
    """Protocol for control flow logic."""
    
    async def execute(
        self, 
        modules: Dict[str, LanguageModule],
        input_data: Dict[str, Any],
        inference_client: Any
    ) -> Dict[str, Any]:
        """Execute the control flow."""
        ...


class SequentialFlow:
    """Simple sequential execution flow."""
    
    def __init__(self, module_order: List[str]):
        self.module_order = module_order
    
    async def execute(
        self,
        modules: Dict[str, LanguageModule],
        input_data: Dict[str, Any],
        inference_client: Any
    ) -> Dict[str, Any]:
        """Execute modules sequentially."""
        current_data = input_data.copy()
        
        for module_id in self.module_order:
            if module_id not in modules:
                raise ValueError(f"Module {module_id} not found")
            
            module = modules[module_id]
            
            # Validate input if schema exists
            if module.input_schema and not module.input_schema.validate(current_data):
                raise ValueError(f"Input validation failed for module {module_id}")
            
            # Execute module
            from ..inference.base import InferenceRequest
            
            # Format prompt with context data
            formatted_prompt = module.prompt
            if 'text' in current_data:
                formatted_prompt = formatted_prompt.replace('{text}', str(current_data['text']))
            
            request = InferenceRequest(
                prompt=formatted_prompt,
                max_tokens=100,
                temperature=0.1
            )
            
            response = await inference_client.generate(request)
            
            # Extract output from response
            output_text = response.text if hasattr(response, 'text') else str(response)
            
            # Update current data with response
            current_data['output'] = output_text.strip()
            current_data[f'{module_id}_output'] = output_text.strip()
        
        return current_data


class CompoundAISystem:
    """
    Compound AI System as defined in GEPA paper.
    
    Φ = (M, C, X, Y) where:
    - M = language modules
    - C = control flow logic  
    - X = global input schema
    - Y = global output schema
    """
    
    def __init__(
        self,
        modules: Dict[str, LanguageModule],
        control_flow: ControlFlow,
        input_schema: IOSchema,
        output_schema: IOSchema,
        system_id: Optional[str] = None
    ):
        self.modules = modules
        self.control_flow = control_flow
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.system_id = system_id or "default"
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        inference_client: Any
    ) -> Dict[str, Any]:
        """Execute the compound AI system."""
        # Validate input
        if not self.input_schema.validate(input_data):
            raise ValueError("Input validation failed")
        
        # Execute control flow
        result = await self.control_flow.execute(
            self.modules,
            input_data,
            inference_client
        )
        
        # Validate output
        if not self.output_schema.validate(result):
            raise ValueError("Output validation failed")
        
        return result
    
    def update_module(self, module_id: str, new_prompt: str) -> "CompoundAISystem":
        """Create a new system with updated module prompt."""
        if module_id not in self.modules:
            raise ValueError(f"Module {module_id} not found")
        
        new_modules = self.modules.copy()
        new_modules[module_id] = self.modules[module_id].update_prompt(new_prompt)
        
        return CompoundAISystem(
            modules=new_modules,
            control_flow=self.control_flow,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            system_id=self.system_id
        )
    
    def get_module_prompts(self) -> Dict[str, str]:
        """Get all module prompts."""
        return {mid: module.prompt for mid, module in self.modules.items()}
    
    def clone(self) -> "CompoundAISystem":
        """Create a deep copy of the system."""
        return CompoundAISystem(
            modules=self.modules.copy(),
            control_flow=self.control_flow,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            system_id=self.system_id
        )