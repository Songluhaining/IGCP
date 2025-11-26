package ast.c.preprocessor.blockstarter;

import ast.walking.ASTNodeVisitor;

public class PreIfStatement extends PreElIfStatement {

	private PreElIfStatement elIfNode = null;


	@Override
	public void accept(ASTNodeVisitor visitor) {
		visitor.visit(this);
	}
	
	public PreElIfStatement getElIfNode() {
		return elIfNode;
	}

	public void setElIfNode(PreElIfStatement elIfNode) {
		this.elIfNode = elIfNode;
	}

}
