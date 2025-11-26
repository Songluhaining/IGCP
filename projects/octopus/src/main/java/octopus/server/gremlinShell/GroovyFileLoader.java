package octopus.server.gremlinShell;

import java.io.IOException;
import java.nio.file.Path;
import org.codehaus.groovy.control.CompilationFailedException;
import groovy.lang.GroovyShell;
import octopus.server.gremlinShell.fileWalker.SourceFileListener;

public class GroovyFileLoader extends SourceFileListener {

	private GroovyShell groovyShell;

	public void setGroovyShell(GroovyShell groovyshell) {
		this.groovyShell = groovyshell;
	}

	@Override
	public void visitFile(Path filename) {
		try {
			groovyShell.evaluate(filename.toFile());
		} catch (CompilationFailedException e) {
			System.out.println("Compilation failure for standard library: "+ e.getMessage());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public void initialize() {
	}

	@Override
	public void shutdown() {
	}

	@Override
	public void preVisitDirectory(Path dir) {
	}

	@Override
	public void postVisitDirectory(Path dir) {
	}

}
