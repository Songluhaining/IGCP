package joern.plugins.importer;

import java.io.IOException;
import org.json.JSONObject;
import fileWalker.OrderedWalker;
import joern.api.JoernProject;
import joern.api.plugintypes.JoernProjectPlugin;

public class JoernImporter extends JoernProjectPlugin {

	private boolean parsecode = true;
	private boolean importcsv = true;
	String projectPath;

	private JoernProject joernProject;

	@Override
	public void configure(JSONObject settings) {
		super.configure(settings);

		if (settings.has("noparsecode"))
			parsecode = false;
		if (settings.has("noimportcsv"))
			importcsv = false;
		
		if (settings.has("srcDir")) {		
			projectPath = settings.getString("srcDir");
		}
		else {
			System.out.println("No source found!");
		}
	}

	@Override
	public void execute() throws Exception {
		openProject();
		if (parsecode)
			parseSourceCode();
		if (importcsv)
			importCSVFilesIntoDatabase();		
	}

	private void openProject() {
		joernProject = (JoernProject) getProjectConnector().getWrapper();
	}


	private void parseSourceCode() {
		System.out.println("Parsing code");

		String parserOutputDirectory = joernProject.getParserOutputDirectory();
		//This path is not on the server
		String sourceCodeDirectory = projectPath;

		CParserWrapper parserWrapper = new CParserWrapper();
		parserWrapper.setMultiFileOutput(false);
		parserWrapper.initialize(parserOutputDirectory);
		parserWrapper.walkCodebase(new String[] { sourceCodeDirectory });

		System.out.println("Parsing complete");
	}

	private void importCSVFilesIntoDatabase() throws IOException {
		System.out.println("Importing graph");

		String parserOutputDirectory = joernProject.getParserOutputDirectory();

		OrderedWalker walker = new OrderedWalker();
		walker.setFilenameFilter("*nodes.csv");
		ImporterListener listener = new ImporterListener();
		listener.setProject(joernProject);

		walker.addListener(listener);
		walker.walk(new String[] { parserOutputDirectory });

		System.out.println("Import complete");
	}

}
