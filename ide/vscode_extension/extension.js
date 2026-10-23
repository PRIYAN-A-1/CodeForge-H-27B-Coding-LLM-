/**
 * CodeForge AI - Visual Studio Code Extension.
 * Integrates autonomous project generation, sandbox validation, and self-repair
 * directly into developer editor workflows.
 */

const vscode = require('vscode');
const http = require('http');

function getApiUrl() {
  const config = vscode.workspace.getConfiguration('codeforge');
  return config.get('apiUrl') || 'http://127.0.0.1:8000';
}

function postRequest(endpoint, payload) {
  return new Promise((resolve, reject) => {
    const url = new URL(endpoint, getApiUrl());
    const data = JSON.stringify(payload);

    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data),
      },
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => (body += chunk));
      res.on('end', () => {
        try {
          const parsed = JSON.parse(body);
          if (res.statusCode >= 400) {
            reject(new Error(parsed.detail || `HTTP Error ${res.statusCode}`));
          } else {
            resolve(parsed);
          }
        } catch (e) {
          resolve(body);
        }
      });
    });

    req.on('error', (e) => reject(e));
    req.write(data);
    req.end();
  });
}

function activate(context) {
  // 1. Generate Full-Stack Project Command
  const genCmd = vscode.commands.registerCommand('codeforge.generateProject', async () => {
    const prompt = await vscode.window.showInputBox({
      prompt: 'Describe the software project you want to build autonomously:',
      placeHolder: 'e.g. Build an expense management app with Next.js, FastAPI, and PostgreSQL',
      validateInput: (text) => (text && text.trim().length >= 5 ? null : 'Requirement must be at least 5 characters.'),
    });

    if (!prompt) return;

    vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: 'CodeForge AI: Synthesizing Project...',
        cancellable: false,
      },
      async (progress) => {
        try {
          progress.report({ increment: 20, message: 'Analyzing requirements & stack...' });
          const plan = await postRequest('/api/projects/plan', { requirement: prompt });

          progress.report({ increment: 40, message: `Synthesizing code for ${plan.project_name}...` });
          const gen = await postRequest('/api/projects/generate', plan);

          progress.report({ increment: 30, message: 'Running isolated sandbox tests...' });
          const val = await postRequest(`/api/projects/${gen.project_id}/validate`, {});

          progress.report({ increment: 10, message: 'Finalizing bundle...' });

          const statusMsg = val.all_passed
            ? `Successfully generated ${gen.total_files} files with 100% passing tests!`
            : `Generated ${gen.total_files} files. ${val.unresolved_errors.length} test issues detected.`;

          vscode.window.showInformationMessage(
            `CodeForge AI: ${statusMsg} Project ID: ${gen.project_id}`,
            'Open Dashboard',
            'View Files'
          ).then((selection) => {
            if (selection === 'Open Dashboard') {
              vscode.env.openExternal(vscode.Uri.parse(getApiUrl() + '/dashboard'));
            }
          });
        } catch (err) {
          vscode.window.showErrorMessage(`CodeForge AI Error: ${err.message}`);
        }
      }
    );
  });

  // 2. Validate Project Command
  const valCmd = vscode.commands.registerCommand('codeforge.validateProject', async () => {
    const projectId = await vscode.window.showInputBox({
      prompt: 'Enter Project ID to validate in sandbox:',
      placeHolder: 'proj-xxxx',
    });
    if (!projectId) return;

    try {
      const val = await postRequest(`/api/projects/${projectId}/validate`, {});
      if (val.all_passed) {
        vscode.window.showInformationMessage(`Validation PASSED: All ${val.tests_passed} checks verified.`);
      } else {
        vscode.window.showWarningMessage(`Validation FAILED: ${val.unresolved_errors.join('; ')}`);
      }
    } catch (err) {
      vscode.window.showErrorMessage(`Validation Error: ${err.message}`);
    }
  });

  // 3. Repair Project Command
  const repCmd = vscode.commands.registerCommand('codeforge.repairProject', async () => {
    const projectId = await vscode.window.showInputBox({
      prompt: 'Enter Project ID to repair:',
      placeHolder: 'proj-xxxx',
    });
    if (!projectId) return;

    vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `CodeForge AI: Self-repairing ${projectId}...`,
        cancellable: false,
      },
      async () => {
        try {
          const outcome = await postRequest(`/api/projects/${projectId}/repair`, {
            project_id: projectId,
            errors: ['Syntax or test failure in backend'],
            max_attempts: 3,
          });
          if (outcome.success) {
            vscode.window.showInformationMessage(
              `Repair SUCCEEDED: Fixed in ${outcome.attempts_used} attempts.`
            );
          } else {
            vscode.window.showErrorMessage(
              `Repair FAILED: Exceeded ${outcome.max_attempts} attempts.`
            );
          }
        } catch (err) {
          vscode.window.showErrorMessage(`Repair Error: ${err.message}`);
        }
      }
    );
  });

  // 4. Open Web Dashboard Command
  const dashCmd = vscode.commands.registerCommand('codeforge.openDashboard', () => {
    vscode.env.openExternal(vscode.Uri.parse(getApiUrl() + '/dashboard'));
  });

  context.subscriptions.push(genCmd, valCmd, repCmd, dashCmd);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
