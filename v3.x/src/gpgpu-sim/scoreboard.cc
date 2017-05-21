// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "scoreboard.h"
#include "shader.h"
#include "../cuda-sim/ptx_sim.h"
#include "shader_trace.h"


//Constructor
Scoreboard::Scoreboard( unsigned sid, unsigned n_warps )
: longopregs()
{
	m_sid = sid;
	//Initialize size of table
	reg_table.resize(n_warps);
	longopregs.resize(n_warps);
}

// Print scoreboard contents
void Scoreboard::printContents() const
{
	printf("scoreboard contents (sid=%d): \n", m_sid);
	for(unsigned i=0; i<reg_table.size(); i++) {
		if(reg_table[i].size() == 0 ) continue;
		printf("  wid = %2d: ", i);
		std::set<unsigned>::const_iterator it;
		for( it=reg_table[i].begin() ; it != reg_table[i].end(); it++ )
			printf("%u ", *it);
		printf("\n");
	}
}

void Scoreboard::reserveRegister(unsigned wid, unsigned regnum) 
{
	if( !(reg_table[wid].find(regnum) == reg_table[wid].end()) ){
		printf("Error: trying to reserve an already reserved register (sid=%d, wid=%d, regnum=%d).", m_sid, wid, regnum);
        abort();
	}
    SHADER_DPRINTF( SCOREBOARD,
                    "Reserved Register - warp:%d, reg: %d\n", wid, regnum );
	reg_table[wid].insert(regnum);
}

// Unmark register as write-pending
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum) 
{
	if( !(reg_table[wid].find(regnum) != reg_table[wid].end()) ) 
        return;
    SHADER_DPRINTF( SCOREBOARD,
                    "Release register - warp:%d, reg: %d\n", wid, regnum );
	reg_table[wid].erase(regnum);
}

const bool Scoreboard::islongop (unsigned warp_id,unsigned regnum) {
	return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

void Scoreboard::reserveRegisters(const class warp_inst_t* inst) 
{
    for( unsigned r=0; r < 4; r++) {
        if(inst->out[r] > 0) {
            reserveRegister(inst->warp_id(), inst->out[r]);
            SHADER_DPRINTF( SCOREBOARD,
                            "Reserved register - warp:%d, reg: %d\n",
                            inst->warp_id(),
                            inst->out[r] );
        }
    }

    //Keep track of long operations
    if (inst->is_load() &&
    		(	inst->space.get_type() == global_space ||
    			inst->space.get_type() == local_space ||
                inst->space.get_type() == param_space_kernel ||
                inst->space.get_type() == param_space_local ||
                inst->space.get_type() == param_space_unclassified ||
    			inst->space.get_type() == tex_space)){
    	for ( unsigned r=0; r<4; r++) {
    		if(inst->out[r] > 0) {
                SHADER_DPRINTF( SCOREBOARD,
                                "New longopreg marked - warp:%d, reg: %d\n",
                                inst->warp_id(),
                                inst->out[r] );
                longopregs[inst->warp_id()].insert(inst->out[r]);
            }
    	}
    }
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(const class warp_inst_t *inst) 
{
    for( unsigned r=0; r < 4; r++) {
        if(inst->out[r] > 0) {
            SHADER_DPRINTF( SCOREBOARD,
                            "Register Released - warp:%d, reg: %d\n",
                            inst->warp_id(),
                            inst->out[r] );
            releaseRegister(inst->warp_id(), inst->out[r]);
            longopregs[inst->warp_id()].erase(inst->out[r]);
        }
    }
}

/** 
 * Checks to see if registers used by an instruction are reserved in the scoreboard
 *  
 * @return 
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/ 
bool Scoreboard::checkCollision( unsigned wid, const class inst_t *inst ) const
{
	// Get list of all input and output registers
	std::set<int> inst_regs;

	if(inst->out[0] > 0) inst_regs.insert(inst->out[0]);
	if(inst->out[1] > 0) inst_regs.insert(inst->out[1]);
	if(inst->out[2] > 0) inst_regs.insert(inst->out[2]);
	if(inst->out[3] > 0) inst_regs.insert(inst->out[3]);
	if(inst->in[0] > 0) inst_regs.insert(inst->in[0]);
	if(inst->in[1] > 0) inst_regs.insert(inst->in[1]);
	if(inst->in[2] > 0) inst_regs.insert(inst->in[2]);
	if(inst->in[3] > 0) inst_regs.insert(inst->in[3]);
	if(inst->pred > 0) inst_regs.insert(inst->pred);
	if(inst->ar1 > 0) inst_regs.insert(inst->ar1);
	if(inst->ar2 > 0) inst_regs.insert(inst->ar2);

	// Check for collision, get the intersection of reserved registers and instruction registers
	std::set<int>::const_iterator it2;
	for ( it2=inst_regs.begin() ; it2 != inst_regs.end(); it2++ )
		if(reg_table[wid].find(*it2) != reg_table[wid].end()) {
			return true;
		}
	return false;
}

// @JD
// Return T if current inst has next inst dependent and can be issued in the chain
bool Scoreboard::checkdependencyRegister( unsigned wid1, const class inst_t *inst1, const class inst_t *inst2) const
{
//     Only resolve dependencies in the same warp
//    if (wid1 != wid2){
//        return false;
//    }

    // check for dependency detected flags raised
    if(inst1->)

    // Get list of all input and output registers from both instructions
    std::set<int> inst1_regs;
    std::set<int> inst2_regs;

//    if(inst1->out[0] > 0) inst1_regs.insert(inst1->out[0]);
//    if(inst1->out[1] > 0) inst1_regs.insert(inst1->out[1]);
//    if(inst1->out[2] > 0) inst1_regs.insert(inst1->out[2]);
//    if(inst1->out[3] > 0) inst1_regs.insert(inst1->out[3]);
//    if(inst1->in[0] > 0) inst1_regs.insert(inst1->in[0]);
//    if(inst1->in[1] > 0) inst1_regs.insert(inst1->in[1]);
//    if(inst1->in[2] > 0) inst1_regs.insert(inst1->in[2]);
//    if(inst1->in[3] > 0) inst1_regs.insert(inst1->in[3]);
//    if(inst1->pred > 0) inst1_regs.insert(inst1->pred);
//    if(inst1->ar1 > 0) inst1_regs.insert(inst1->ar1);
//    if(inst1->ar2 > 0) inst1_regs.insert(inst1->ar2);
//
//    if(inst2->out[0] > 0) inst2_regs.insert(inst2->out[0]);
//    if(inst2->out[1] > 0) inst2_regs.insert(inst2->out[1]);
//    if(inst2->out[2] > 0) inst2_regs.insert(inst2->out[2]);
//    if(inst2->out[3] > 0) inst2_regs.insert(inst2->out[3]);
//    if(inst2->in[0] > 0) inst2_regs.insert(inst2->in[0]);
//    if(inst2->in[1] > 0) inst2_regs.insert(inst2->in[1]);
//    if(inst2->in[2] > 0) inst2_regs.insert(inst2->in[2]);
//    if(inst2->in[3] > 0) inst2_regs.insert(inst2->in[3]);
//    if(inst2->pred > 0) inst2_regs.insert(inst2->pred);
//    if(inst2->ar1 > 0) inst2_regs.insert(inst2->ar1);
//    if(inst2->ar2 > 0) inst2_regs.insert(inst2->ar2);

    // Search for RAW hazard inst1 write reg read by inst2 read operand
    std::set<int>::const_iterator it1;
    for ( it1=inst1_regs.begin() ; it1 != inst1_regs.end(); it1++ )
            if(reg_table[wid1].find(*it1) != reg_table[wid1].end()) {
            return true;
        }
}

// @JD
// Func to check if operand in other chained instruction is not stalled due to another data dependency
bool Scoreboard::checkpartialCollision(unsigned int wid, const class inst_t *inst, unsigned int *reg_index)
{
    std::set<int> inst_regs;

    for(int j = 0; j<MAX_REG_OPERANDS/4; j++)
    {
        if(inst->in[reg_index[j]] > 0){
            inst_regs.insert(inst->in[j]);
        }
    }

    // to check if remaining operands of the dependent inst are free
    std::set<int>::const_iterator it;
    for ( it=inst_regs.begin() ; it != inst_regs.end(); it++ )
        if(reg_table[wid].find(*it) != reg_table[wid].end()) {
            return true;
        }
    return false;
}

bool Scoreboard::pendingWrites(unsigned wid) const
{
	return !reg_table[wid].empty();
}
