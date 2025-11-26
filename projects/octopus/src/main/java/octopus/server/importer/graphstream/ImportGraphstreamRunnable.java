package octopus.server.importer.graphstream;

import java.io.IOException;
import octopus.api.database.Database;
import octopus.api.projects.OctopusProject;
import octopus.api.projects.ProjectManager;
import octopus.server.importer.graphstream.titan.GraphstreamImporter;

public class ImportGraphstreamRunnable implements Runnable {

	private final ImportJob importJob;

	public ImportGraphstreamRunnable(ImportJob importJob) {
		this.importJob = importJob;
	}

	@Override
	public void run() {
		System.out.println("Starting graph stream import");

		GraphstreamImporter gdsBatchImporter = new GraphstreamImporter();

		String streamFilename = importJob.getStreamFilename();
		String projectName = importJob.getProjectName();

		ProjectManager projectManager = new ProjectManager();
		OctopusProject project = projectManager.getProjectByName(projectName);
		if (project == null)
			throw new RuntimeException("Error: project dos not exist");

		try {
			Database database = project.getNewDatabaseInstance();
			gdsBatchImporter.setGraph(database.getGraph());
			gdsBatchImporter.importGraphstreamFiles(streamFilename);
			database.closeInstance();
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println("Graph stream import finished");
	}

}
