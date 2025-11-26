package ast.c.preprocessor.blockstarter;

import ast.preprocessor.PreBlockstarter;
import ast.walking.ASTNodeVisitor;

public class PreElseStatement extends PreBlockstarter {
	
	private PreEndIfStatement endIfNode = null;
	
	@Override
	public void accept(ASTNodeVisitor visitor) {
		visitor.visit(this);
	}
	

	public PreEndIfStatement getEndIfNode() {
		return endIfNode;
	}

	public void setEndIfNode(PreEndIfStatement endIfNode) {
		this.endIfNode = endIfNode;
	}
}
