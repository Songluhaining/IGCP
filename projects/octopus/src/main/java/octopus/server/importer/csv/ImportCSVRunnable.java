package octopus.server.importer.csv;

import java.io.IOException;
import octopus.api.database.Database;
import octopus.api.projects.OctopusProject;
import octopus.api.projects.ProjectManager;
import octopus.server.importer.csv.titan.CSVImporter;

public class ImportCSVRunnable implements Runnable {

	private final ImportJob importJob;

	public ImportCSVRunnable(ImportJob importJob) {
		this.importJob = importJob;
	}

	@Override
	public void run() {
		System.out.println("Starting CSV import");
		
		
		CSVImporter csvBatchImporter = new CSVImporter();

		String nodeFilename = importJob.getNodeFilename();
		String edgeFilename = importJob.getEdgeFilename();
		String projectName = importJob.getProjectName();

		ProjectManager projectManager = new ProjectManager();
		OctopusProject project = projectManager.getProjectByName(projectName);
		
		if (project == null)
			throw new RuntimeException("Error: project does not exist");

		try {
			Database database = project.getNewDatabaseInstance();
			csvBatchImporter.setGraph(database.getGraph());
			csvBatchImporter.importCSVFiles(nodeFilename, edgeFilename);
			database.closeInstance();
			
		
		} catch (IOException e) {
			System.err.println("Error in ImportCSVRunnable");
			e.printStackTrace();
		}

		System.out.println("CSV import finished");
	}

}
