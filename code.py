import random
from enum import Enum
import numpy as np
from collections import OrderedDict, deque

class InstructionType(Enum):
    R_TYPE = 1
    I_TYPE = 2
    S_TYPE = 3
    B_TYPE = 4
    U_TYPE = 5
    J_TYPE = 6
    DL_OP = 7

class Instruction:
    def __init__(self, opcode, rs1, rs2, rd, imm=None, func3=None, func7=None):
        self.opcode = opcode
        self.rs1 = rs1
        self.rs2 = rs2
        self.rd = rd
        self.imm = imm
        self.func3 = func3
        self.func7 = func7
        self.result = None
        self.cycles = 0

    def __str__(self):
        return f"OP: {self.opcode}, RS1: {self.rs1}, RS2: {self.rs2}, RD: {self.rd}, IMM: {self.imm}, FUNC3: {self.func3}, FUNC7: {self.func7}"

class Cache:
    def __init__(self, size, block_size, associativity):
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.num_sets = size // (block_size * associativity)
        self.cache = [OrderedDict() for _ in range(self.num_sets)]
        self.hits = 0
        self.misses = 0

    def access(self, address):
        set_index = (address // self.block_size) % self.num_sets
        tag = address // (self.block_size * self.num_sets)
        
        if tag in self.cache[set_index]:
            self.cache[set_index].move_to_end(tag)
            self.hits += 1
            return True
        else:
            if len(self.cache[set_index]) >= self.associativity:
                self.cache[set_index].popitem(last=False)
            self.cache[set_index][tag] = True
            self.misses += 1
            return False

class BranchPredictor:
    def __init__(self, size=1024):
        self.table = [1] * size  # Initialize with Weakly Taken
        self.size = size

    def predict(self, pc):
        index = pc % self.size
        return self.table[index] >= 2

    def update(self, pc, taken):
        index = pc % self.size
        if taken:
            self.table[index] = min(3, self.table[index] + 1)
        else:
            self.table[index] = max(0, self.table[index] - 1)

class DLAccelerator:
    def __init__(self, scratchpad_size=1024):
        self.scratchpad = [0] * scratchpad_size
        self.scratchpad_size = scratchpad_size

    def execute(self, operation, input1, input2):
        a = np.array(self.scratchpad[input1:input1+16]).reshape((4, 4))
        b = np.array(self.scratchpad[input2:input2+16]).reshape((4, 4))
        
        if operation == 0:
            result = np.matmul(a, b)
            cycles = 20
        elif operation == 1:
            result = np.multiply(a, b)
            cycles = 10
        elif operation == 2:
            result = np.convolve(a.flatten(), b.flatten(), mode='same').reshape((4, 4))
            cycles = 30
        elif operation == 3:
            result = np.maximum(a, 0)
            cycles = 5
        elif operation == 4:
            result = np.maximum.reduce([a[::2, ::2], a[1::2, ::2], a[::2, 1::2], a[1::2, 1::2]])
            cycles = 15
        else:
            result = np.zeros((4, 4), dtype=np.int32)
            cycles = 1
        
        return int(np.sum(result) % (2**32)), cycles

class Pipeline:
    def __init__(self, memory_size=1024):
        self.memory_size = max(memory_size, 1024)
        self.registers = [0] * 32
        self.memory = [0] * self.memory_size
        self.pc = 0
        self.cycle_count = 0
        self.instruction_count = 0
        self.icache = Cache(size=1024, block_size=64, associativity=2)
        self.dcache = Cache(size=1024, block_size=64, associativity=2)
        self.branch_predictor = BranchPredictor()
        self.dl_accelerator = DLAccelerator()
        self.pipeline_stages = {
            'fetch': None,
            'decode': None,
            'execute': None,
            'memory': None,
            'writeback': None
        }
        self.reorder_buffer = deque(maxlen=16)
        self.mispredictions = 0
        self.stalls = 0
        self.data_hazards = 0

    def fetch(self):
        if self.pc < 0 or self.pc >= self.memory_size:
            print(f"Warning: PC out of bounds. PC: {self.pc}")
            return None

        if self.icache.access(self.pc):
            instruction = self.memory[self.pc]
            self.pc += 1
            return instruction
        else:
            instruction = self.memory[self.pc]
            self.pc += 1
            self.stalls += 9  # Additional 9 cycles for cache miss
            return instruction

    def decode(self, instruction):
        if instruction is None:
            return None
        rs1_value = self.registers[instruction.rs1]
        rs2_value = self.registers[instruction.rs2]
        instruction.rs1_value = rs1_value
        instruction.rs2_value = rs2_value
        return instruction

    def execute(self, instruction):
        if instruction is None:
            return None

        old_pc = self.pc  # Store old PC for debugging
        result = None
        cycles = 1

        if instruction.opcode == InstructionType.R_TYPE:
            if instruction.func3 == 0x0 and instruction.func7 == 0x00:  # ADD
                result = instruction.rs1_value + instruction.rs2_value
            elif instruction.func3 == 0x0 and instruction.func7 == 0x20:  # SUB
                result = instruction.rs1_value - instruction.rs2_value
            elif instruction.func3 == 0x7 and instruction.func7 == 0x00:  # AND
                result = instruction.rs1_value & instruction.rs2_value
            elif instruction.func3 == 0x6 and instruction.func7 == 0x00:  # OR
                result = instruction.rs1_value | instruction.rs2_value
            elif instruction.func3 == 0x4 and instruction.func7 == 0x00:  # XOR
                result = instruction.rs1_value ^ instruction.rs2_value
            elif instruction.func3 == 0x1 and instruction.func7 == 0x00:  # SLL
                result = instruction.rs1_value << (instruction.rs2_value & 0x1F)
            elif instruction.func3 == 0x5 and instruction.func7 == 0x00:  # SRL
                result = instruction.rs1_value >> (instruction.rs2_value & 0x1F)
            elif instruction.func3 == 0x5 and instruction.func7 == 0x20:  # SRA
                result = instruction.rs1_value >> (instruction.rs2_value & 0x1F) if instruction.rs1_value >= 0 else ~(~instruction.rs1_value >> (instruction.rs2_value & 0x1F))
            else:
                result = 0  # Unsupported R-type instruction
        elif instruction.opcode == InstructionType.I_TYPE:
            if instruction.func3 == 0x0:  # ADDI
                result = instruction.rs1_value + instruction.imm
            elif instruction.func3 == 0x7:  # ANDI
                result = instruction.rs1_value & instruction.imm
            elif instruction.func3 == 0x6:  # ORI
                result = instruction.rs1_value | instruction.imm
            elif instruction.func3 == 0x4:  # XORI
                result = instruction.rs1_value ^ instruction.imm
            elif instruction.func3 == 0x1:  # SLLI
                result = instruction.rs1_value << (instruction.imm & 0x1F)
            elif instruction.func3 == 0x5 and (instruction.imm & 0xF00) == 0x000:  # SRLI
                result = instruction.rs1_value >> (instruction.imm & 0x1F)
            elif instruction.func3 == 0x5 and (instruction.imm & 0xF00) == 0x400:  # SRAI
                result = instruction.rs1_value >> (instruction.imm & 0x1F) if instruction.rs1_value >= 0 else ~(~instruction.rs1_value >> (instruction.imm & 0x1F))
            else:
                result = 0  # Unsupported I-type instruction
        elif instruction.opcode == InstructionType.S_TYPE:
            address = instruction.rs1_value + instruction.imm
            if 0 <= address < self.memory_size:
                if self.dcache.access(address):
                    cycles = 1
                else:
                    cycles = 10
            else:
                print(f"Warning: Memory access out of bounds. Address: {address}")
                cycles = 1
            result = None
        elif instruction.opcode == InstructionType.B_TYPE:
            if instruction.func3 == 0x0:  # BEQ
                taken = instruction.rs1_value == instruction.rs2_value
            elif instruction.func3 == 0x1:  # BNE
                taken = instruction.rs1_value != instruction.rs2_value
            elif instruction.func3 == 0x4:  # BLT
                taken = instruction.rs1_value < instruction.rs2_value
            elif instruction.func3 == 0x5:  # BGE
                taken = instruction.rs1_value >= instruction.rs2_value
            elif instruction.func3 == 0x6:  # BLTU
                taken = (instruction.rs1_value & 0xFFFFFFFF) < (instruction.rs2_value & 0xFFFFFFFF)
            elif instruction.func3 == 0x7:  # BGEU
                taken = (instruction.rs1_value & 0xFFFFFFFF) >= (instruction.rs2_value & 0xFFFFFFFF)
            else:
                taken = False

            predicted = self.branch_predictor.predict(old_pc)
            self.branch_predictor.update(old_pc, taken)

            if predicted != taken:
                self.mispredictions += 1
                cycles = 3
                if taken:
                    new_pc = old_pc + instruction.imm
                    if 0 <= new_pc < self.memory_size:
                        print(f"Branch mispredicted (taken): Old PC: {old_pc}, New PC: {new_pc}, Imm: {instruction.imm}")
                        self.pc = new_pc
                    else:
                        print(f"Warning: Branch target out of bounds. Old PC: {old_pc}, Target PC: {new_pc}")
                        self.pc = old_pc + 1  # Move to next instruction instead
                else:
                    print(f"Branch mispredicted (not taken): PC: {self.pc}")
            else:
                if taken:
                    new_pc = old_pc + instruction.imm
                    if 0 <= new_pc < self.memory_size:
                        print(f"Branch correctly predicted (taken): Old PC: {old_pc}, New PC: {new_pc}, Imm: {instruction.imm}")
                        self.pc = new_pc
                    else:
                        print(f"Warning: Branch target out of bounds. Old PC: {old_pc}, Target PC: {new_pc}")
                        self.pc = old_pc + 1  # Move to next instruction instead
                else:
                    print(f"Branch correctly predicted (not taken): PC: {self.pc}")

            result = None
        elif instruction.opcode == InstructionType.U_TYPE:
            result = instruction.imm << 12
        elif instruction.opcode == InstructionType.J_TYPE:
            result = self.pc
            new_pc = old_pc + instruction.imm
            print(f"Jump: Old PC: {old_pc}, New PC: {new_pc}, Imm: {instruction.imm}")
            self.pc = new_pc
        elif instruction.opcode == InstructionType.DL_OP:
            result, cycles = self.dl_accelerator.execute(instruction.func3, instruction.rs1_value, instruction.rs2_value)
        else:
            result = 0

        instruction.result = result
        instruction.cycles = cycles
        return instruction

    def memory_stage(self, instruction):
        if instruction is None or instruction.opcode != InstructionType.S_TYPE:
            return instruction

        address = instruction.rs1_value + instruction.imm
        if 0 <= address < self.memory_size:
            self.memory[address] = instruction.rs2_value & 0xFFFFFFFF
        else:
            print(f"Warning: Memory access out of bounds in memory stage. Address: {address}")
        return instruction

    def writeback(self, instruction):
        if instruction is None:
            return
        if instruction.opcode not in [InstructionType.S_TYPE, InstructionType.B_TYPE] and instruction.rd != 0:
            self.registers[instruction.rd] = instruction.result & 0xFFFFFFFF
        self.instruction_count += 1

    def run(self, instructions, max_cycles):
        # Initialize memory with instructions
        for i, inst in enumerate(instructions):
            if i < self.memory_size:
                self.memory[i] = inst
            else:
                print(f"Warning: Not all instructions fit in memory. Truncating at instruction {i}.")
                break
        self.pc = 0

        while self.cycle_count < max_cycles:
            print(f"Cycle {self.cycle_count}: PC = {self.pc}")
            self.cycle_count += 1
            cycle_stalls = 0

            # Writeback Stage
            if self.pipeline_stages['writeback']:
                self.writeback(self.pipeline_stages['writeback'])
                if self.reorder_buffer:
                    self.reorder_buffer.popleft()

            # Memory Stage
            self.pipeline_stages['writeback'] = self.memory_stage(self.pipeline_stages['memory'])
            
            # Execute Stage
            self.pipeline_stages['memory'] = self.execute(self.pipeline_stages['execute'])
            if self.pipeline_stages['memory']:
                cycle_stalls += self.pipeline_stages['memory'].cycles - 1

            # Decode Stage
            self.pipeline_stages['execute'] = self.decode(self.pipeline_stages['decode'])

            # Fetch Stage
            if len(self.reorder_buffer) < 16:
                fetched_instruction = self.fetch()
                if fetched_instruction:
                    self.pipeline_stages['decode'] = fetched_instruction
                    self.reorder_buffer.append(fetched_instruction)
                else:
                    self.pipeline_stages['decode'] = None
                    cycle_stalls += 1
            else:
                cycle_stalls += 1
                self.pipeline_stages['decode'] = None

            # Update stalls
            self.stalls += cycle_stalls
            self.cycle_count += cycle_stalls

            # Check for data hazards
            if self.pipeline_stages['decode'] and self.pipeline_stages['execute']:
                if ((self.pipeline_stages['decode'].rs1 == self.pipeline_stages['execute'].rd or 
                    self.pipeline_stages['decode'].rs2 == self.pipeline_stages['execute'].rd) and 
                    self.pipeline_stages['execute'].rd != 0):
                    self.data_hazards += 1

            # Debug output
            print(f"  Fetch: {self.pipeline_stages['decode']}")
            print(f"  Decode: {self.pipeline_stages['execute']}")
            print(f"  Execute: {self.pipeline_stages['memory']}")
            print(f"  Memory: {self.pipeline_stages['writeback']}")
            print(f"  Writeback: {self.pipeline_stages['writeback']}")
            print(f"  Stalls this cycle: {cycle_stalls}")
            print(f"  Total stalls: {self.stalls}")
            print(f"  Data hazards: {self.data_hazards}")
            print(f"  Reorder buffer size: {len(self.reorder_buffer)}")

            # Check if pipeline is empty
            if all(stage is None for stage in self.pipeline_stages.values()) and len(self.reorder_buffer) == 0:
                break

        return self.cycle_count, self.instruction_count, self.mispredictions, self.stalls, self.data_hazards


def generate_instructions(num_instructions):
    instructions = []
    for _ in range(num_instructions):
        opcode = random.choice(list(InstructionType))
        rs1 = random.randint(0, 31)
        rs2 = random.randint(0, 31)
        rd = random.randint(0, 31)
        imm = random.randint(-2048, 2047)
        func3 = random.randint(0, 7)
        func7 = random.choice([0x00, 0x20])
        instructions.append(Instruction(opcode, rs1, rs2, rd, imm, func3, func7))
    return instructions

def run_simulation(num_instructions, max_cycles):
    try:
        memory_size = max(1024, num_instructions * 2)
        pipeline = Pipeline(memory_size=memory_size)
        instructions = generate_instructions(num_instructions)
        
        cycles, completed_instructions, mispredictions, stalls, data_hazards = pipeline.run(instructions, max_cycles)
        
        print(f"Simulation completed in {cycles} cycles")
        print(f"Instructions completed: {completed_instructions}")
        print(f"Instructions per cycle (IPC): {completed_instructions / cycles:.2f}")
        print(f"Branch mispredictions: {mispredictions}")
        print(f"Stalls: {stalls}")
        print(f"Data hazards: {data_hazards}")
        print(f"Instruction cache - Hits: {pipeline.icache.hits}, Misses: {pipeline.icache.misses}")
        print(f"Data cache - Hits: {pipeline.dcache.hits}, Misses: {pipeline.dcache.misses}")
        
        return cycles, completed_instructions, mispredictions, stalls, data_hazards, pipeline, instructions
    except Exception as e:
        print(f"An error occurred during simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0, 0, None, None

def analyze_results(cycles, completed_instructions, mispredictions, stalls, data_hazards, pipeline, instructions):
    if pipeline and instructions:
        ipc = completed_instructions / cycles if cycles > 0 else 0
        branch_accuracy = 1 - (mispredictions / completed_instructions) if completed_instructions > 0 else 0
        stall_rate = stalls / cycles if cycles > 0 else 0
        data_hazard_rate = data_hazards / completed_instructions if completed_instructions > 0 else 0
        
        icache_hit_rate = pipeline.icache.hits / (pipeline.icache.hits + pipeline.icache.misses) if (pipeline.icache.hits + pipeline.icache.misses) > 0 else 0
        dcache_hit_rate = pipeline.dcache.hits / (pipeline.dcache.hits + pipeline.dcache.misses) if (pipeline.dcache.hits + pipeline.dcache.misses) > 0 else 0
        
        instruction_types = [inst.opcode for inst in instructions]
        type_counts = {t: instruction_types.count(t) for t in set(instruction_types)}
        
        print("\nPerformance Analysis:")
        print(f"Instructions Per Cycle (IPC): {ipc:.2f}")
        print(f"Branch Prediction Accuracy: {branch_accuracy:.2%}")
        print(f"Stall Rate: {stall_rate:.2%}")
        print(f"Data Hazard Rate: {data_hazard_rate:.2%}")
        print(f"Instruction Cache Hit Rate: {icache_hit_rate:.2%}")
        print(f"Data Cache Hit Rate: {dcache_hit_rate:.2%}")
        
        print("\nInstruction Type Distribution:")
        for itype, count in type_counts.items():
            print(f"{itype.name}: {count} ({count/len(instructions):.2%})")
        
        dl_instructions = type_counts.get(InstructionType.DL_OP, 0)
        print(f"\nDL Accelerator Usage: {dl_instructions} instructions ({dl_instructions/len(instructions):.2%})")
        
        single_cycle_time = len(instructions)
        speedup = single_cycle_time / cycles if cycles > 0 else 0
        print(f"\nSpeedup over single-cycle execution: {speedup:.2f}x")
        
    else:
        print("Simulation failed to complete successfully. Unable to analyze results.")

# Run the simulation
if __name__ == "__main__":
    num_instructions = 1000
    max_cycles = 10000
    sim_results = run_simulation(num_instructions, max_cycles)
    analyze_results(*sim_results)
