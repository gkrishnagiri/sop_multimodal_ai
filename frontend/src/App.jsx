import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = "http://localhost:8015/api";

const STEP_STATUS = {
    NOT_STARTED: "not_started",
    RUNNING: "running",
    COMPLETED: "completed",
    FAILED: "failed",
    SKIPPED: "skipped",
};

const PIPELINE_STEPS = [
    {
        key: "extract_audio",
        label: "Audio",
        fullLabel: "Extract Audio",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/extract-audio`,
    },
    {
        key: "transcribe",
        label: "Transcript",
        fullLabel: "Transcribe Audio",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/transcribe`,
    },
    {
        key: "extract_frames",
        label: "Frames",
        fullLabel: "Extract Frames",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/extract-frames`,
    },
    {
        key: "run_ocr",
        label: "OCR",
        fullLabel: "Run OCR",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/run-ocr`,
    },
    {
        key: "run_diarization",
        label: "Diarization",
        fullLabel: "Run Diarization",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/run-diarization`,
    },
    {
        key: "build_timeline",
        label: "Timeline",
        fullLabel: "Build Timeline",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/build-timeline`,
    },
    {
        key: "detect_activities",
        label: "Activities",
        fullLabel: "Detect Activities",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/detect-activities`,
    },
    {
        key: "refine_activities",
        label: "Refine",
        fullLabel: "Refine Activities",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/refine-activities`,
    },
    {
        key: "generate_sop",
        label: "SOP",
        fullLabel: "Generate SOP",
        method: "POST",
        path: (jobId) => `/jobs/${jobId}/generate-sop`,
    },
];

function emptyStepState() {
    const state = {};

    for (const step of PIPELINE_STEPS) {
        state[step.key] = {
            status: STEP_STATUS.NOT_STARTED,
            error: "",
        };
    }

    return state;
}

function normalizePipelineSteps(job) {
    const state = emptyStepState();

    if (!job?.pipeline_steps || typeof job.pipeline_steps !== "object") {
        return state;
    }

    for (const step of PIPELINE_STEPS) {
        const stepPayload = job.pipeline_steps[step.key];

        if (!stepPayload || typeof stepPayload !== "object") {
            continue;
        }

        state[step.key] = {
            status: stepPayload.status || STEP_STATUS.NOT_STARTED,
            error: stepPayload.error || "",
        };
    }

    return state;
}

function isStepDone(status) {
    return status === STEP_STATUS.COMPLETED || status === STEP_STATUS.SKIPPED;
}

function getStepStatusLabel(status) {
    if (status === STEP_STATUS.COMPLETED) {
        return "Completed";
    }

    if (status === STEP_STATUS.RUNNING) {
        return "Running";
    }

    if (status === STEP_STATUS.FAILED) {
        return "Failed";
    }

    if (status === STEP_STATUS.SKIPPED) {
        return "Skipped";
    }

    return "Pending";
}

function findFirstRunnableStep(stepState) {
    return PIPELINE_STEPS.find(
        (step) => !isStepDone(stepState[step.key]?.status),
    );
}

function getPipelineStatusLabel(status) {
    if (status === "completed") {
        return "Completed";
    }

    if (status === "failed") {
        return "Failed";
    }

    if (status === "in_progress") {
        return "In Progress";
    }

    if (status === "uploaded") {
        return "Uploaded";
    }

    return "Not Started";
}

function getJobDisplayName(job) {
    if (!job) {
        return "Unknown job";
    }

    return (
        job.output_filename ||
        job.filename ||
        job.original_filename ||
        job.video_filename ||
        job.job_id ||
        "Untitled job"
    );
}

function getJobSubText(job) {
    const pieces = [];

    if (job?.job_id) {
        pieces.push(job.job_id);
    }

    if (job?.status) {
        pieces.push(`Backend status: ${job.status}`);
    }

    if (job?.pipeline_failed_step) {
        pieces.push(`Failed at: ${job.pipeline_failed_step}`);
    }

    return pieces.join(" · ");
}

function extractSopText(sopResponse) {
    if (!sopResponse) {
        return "";
    }

    if (typeof sopResponse === "string") {
        return sopResponse;
    }

    if (sopResponse.markdown) {
        return sopResponse.markdown;
    }

    if (sopResponse.sop_markdown) {
        return sopResponse.sop_markdown;
    }

    if (sopResponse.content) {
        return sopResponse.content;
    }

    if (sopResponse.result?.markdown) {
        return sopResponse.result.markdown;
    }

    if (sopResponse.result?.sop_markdown) {
        return sopResponse.result.sop_markdown;
    }

    if (sopResponse.sop?.markdown) {
        return sopResponse.sop.markdown;
    }

    return "SOP generated. Use Open or Download to view the markdown file.";
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, options);

    let payload = null;
    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
        payload = await response.json();
    } else {
        payload = await response.text();
    }

    if (!response.ok) {
        const message =
            typeof payload === "object" && payload !== null
                ? payload.detail || JSON.stringify(payload)
                : payload;

        throw new Error(message || `Request failed with status ${response.status}`);
    }

    return payload;
}

function PipelineStatusBar({ stepState }) {
    return (
        <div className="pipeline-strip">
            {PIPELINE_STEPS.map((step, index) => {
                const status = stepState[step.key]?.status || STEP_STATUS.NOT_STARTED;
                const error = stepState[step.key]?.error || "";

                return (
                    <div className="pipeline-item-wrap" key={step.key}>
                        <div className={`pipeline-item ${status}`}>
                            <div className="step-dot">{index + 1}</div>
                            <div className="step-text">
                                <div className="step-title">{step.label}</div>
                                <div className="step-status-text">
                                    {getStepStatusLabel(status)}
                                </div>
                            </div>
                        </div>

                        {error ? <div className="step-error">{error}</div> : null}
                    </div>
                );
            })}
        </div>
    );
}

export default function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [outputFilename, setOutputFilename] = useState("generated_sop.md");
    const [enableDiarization, setEnableDiarization] = useState(false);
    const [extractScreenshots, setExtractScreenshots] = useState(true);

    const [jobId, setJobId] = useState("");
    const [jobs, setJobs] = useState([]);
    const [selectedJob, setSelectedJob] = useState(null);
    const [selectedJobIds, setSelectedJobIds] = useState([]);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

    const [stepState, setStepState] = useState(emptyStepState());
    const [sopResult, setSopResult] = useState(null);

    const [loadingLabel, setLoadingLabel] = useState("");
    const [errorMessage, setErrorMessage] = useState("");
    const [successMessage, setSuccessMessage] = useState("");

    const canRunJobActions = useMemo(() => Boolean(jobId.trim()), [jobId]);
    const isBusy = Boolean(loadingLabel);

    const selectedJobIdSet = useMemo(
        () => new Set(selectedJobIds),
        [selectedJobIds],
    );

    const selectedJobsForDelete = useMemo(
        () => jobs.filter((job) => selectedJobIdSet.has(job.job_id)),
        [jobs, selectedJobIdSet],
    );

    const hasGeneratedSop =
        stepState.generate_sop?.status === STEP_STATUS.COMPLETED || Boolean(sopResult);

    const failedStep = PIPELINE_STEPS.find(
        (step) => stepState[step.key]?.status === STEP_STATUS.FAILED,
    );

    const nextRunnableStep = findFirstRunnableStep(stepState);

    useEffect(() => {
        document.title = "Agent for Multimodal SOP generation";
        loadJobs();
    }, []);

    async function loadJobs() {
        setErrorMessage("");

        try {
            const payload = await requestJson(`${API_BASE_URL}/jobs`);
            const nextJobs = payload.jobs || [];
            setJobs(nextJobs);
            setSelectedJobIds((currentIds) => {
                const availableIds = new Set(nextJobs.map((job) => job.job_id));
                return currentIds.filter((id) => availableIds.has(id));
            });
        } catch (error) {
            setErrorMessage(error.message);
        }
    }

    async function loadJobById(nextJobId) {
        if (!nextJobId) {
            return null;
        }

        const job = await requestJson(`${API_BASE_URL}/jobs/${nextJobId}`);

        return {
            ...job,
            job_id: job.job_id || nextJobId,
        };
    }

    async function handleUpload(event) {
        event.preventDefault();

        if (!selectedFile) {
            setErrorMessage("Select a video file first.");
            return;
        }

        if (!outputFilename.trim()) {
            setErrorMessage("Output filename is required.");
            return;
        }

        setErrorMessage("");
        setSuccessMessage("");
        setLoadingLabel("Uploading video");
        setSopResult(null);

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            formData.append("output_filename", outputFilename.trim());
            formData.append("enable_diarization", String(enableDiarization));
            formData.append("extract_screenshots", String(extractScreenshots));

            const payload = await requestJson(`${API_BASE_URL}/jobs/upload`, {
                method: "POST",
                body: formData,
            });

            const createdJobId = payload.job_id || payload.id;

            if (!createdJobId) {
                throw new Error("Upload succeeded, but no job_id was returned.");
            }

            const fullJob = await loadJobById(createdJobId);

            setJobId(createdJobId);
            setSelectedJob(fullJob);
            setStepState(normalizePipelineSteps(fullJob));
            setSuccessMessage("Video uploaded. You can now generate the SOP.");

            await loadJobs();
        } catch (error) {
            setErrorMessage(error.message);
        } finally {
            setLoadingLabel("");
        }
    }

    function toggleJobSelection(targetJobId) {
        if (!targetJobId) {
            return;
        }

        setSelectedJobIds((currentIds) => {
            if (currentIds.includes(targetJobId)) {
                return currentIds.filter((id) => id !== targetJobId);
            }

            return [...currentIds, targetJobId];
        });
    }

    function toggleAllJobSelections() {
        if (jobs.length === 0) {
            return;
        }

        setSelectedJobIds((currentIds) => {
            if (currentIds.length === jobs.length) {
                return [];
            }

            return jobs.map((job) => job.job_id).filter(Boolean);
        });
    }

    async function deleteSelectedJobs() {
        if (selectedJobIds.length === 0) {
            return;
        }

        const jobIdsToDelete = [...selectedJobIds];

        setErrorMessage("");
        setSuccessMessage("");
        setShowDeleteConfirm(false);
        setLoadingLabel(`Deleting ${jobIdsToDelete.length} job run${jobIdsToDelete.length === 1 ? "" : "s"}`);

        try {
            const payload = await requestJson(`${API_BASE_URL}/jobs`, {
                method: "DELETE",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ job_ids: jobIdsToDelete }),
            });

            if (jobIdsToDelete.includes(jobId)) {
                setJobId("");
                setSelectedJob(null);
                setStepState(emptyStepState());
                setSopResult(null);
            }

            setSelectedJobIds([]);
            setSuccessMessage(
                `Deleted ${payload.deleted_count || jobIdsToDelete.length} job run${(payload.deleted_count || jobIdsToDelete.length) === 1 ? "" : "s"}.`,
            );
            await loadJobs();
        } catch (error) {
            setErrorMessage(error.message);
        } finally {
            setLoadingLabel("");
        }
    }

    async function selectJob(job) {
        const nextJobId = job.job_id;

        if (!nextJobId) {
            return;
        }

        setErrorMessage("");
        setSuccessMessage("");
        setLoadingLabel("Loading selected job");
        setSopResult(null);

        try {
            const fullJob = await loadJobById(nextJobId);

            setJobId(nextJobId);
            setSelectedJob(fullJob);
            setStepState(normalizePipelineSteps(fullJob));

            if (fullJob.pipeline_status === "completed") {
                await tryLoadSop(nextJobId, false);
            }
        } catch (error) {
            setErrorMessage(error.message);
        } finally {
            setLoadingLabel("");
        }
    }

    async function refreshSelectedJob() {
        if (!jobId.trim()) {
            return;
        }

        try {
            const fullJob = await loadJobById(jobId.trim());
            setSelectedJob(fullJob);
            setStepState(normalizePipelineSteps(fullJob));
        } catch {
            // Ignore background refresh failures.
        }
    }

    async function runSingleStep(step) {
        const currentJobId = jobId.trim();

        if (!currentJobId) {
            throw new Error("No job_id available. Upload a video or select a job.");
        }

        setStepState((currentState) => ({
            ...currentState,
            [step.key]: {
                status: STEP_STATUS.RUNNING,
                error: "",
            },
        }));

        setLoadingLabel(step.fullLabel);

        try {
            const payload = await requestJson(`${API_BASE_URL}${step.path(currentJobId)}`, {
                method: step.method,
            });

            const fullJob = await loadJobById(currentJobId);
            setSelectedJob(fullJob);
            setStepState(normalizePipelineSteps(fullJob));

            if (step.key === "generate_sop") {
                setSopResult(payload);
            }

            return payload;
        } catch (error) {
            await refreshSelectedJob();
            throw error;
        }
    }

    async function generateSopSequentially() {
        if (!jobId.trim()) {
            setErrorMessage("No job_id available. Upload a video or select a job.");
            return;
        }

        setErrorMessage("");
        setSuccessMessage("");
        setSopResult(null);

        try {
            const currentJob = await loadJobById(jobId.trim());
            const currentStepState = normalizePipelineSteps(currentJob);
            const startIndex = PIPELINE_STEPS.findIndex(
                (step) => !isStepDone(currentStepState[step.key]?.status),
            );

            if (startIndex === -1) {
                setSuccessMessage("All pipeline steps are already completed.");
                await tryLoadSop(jobId.trim(), true);
                return;
            }

            for (let index = startIndex; index < PIPELINE_STEPS.length; index += 1) {
                const step = PIPELINE_STEPS[index];
                await runSingleStep(step);
            }

            setSuccessMessage("SOP generated successfully.");
            await loadJobs();
            await tryLoadSop(jobId.trim(), false);
            await refreshSelectedJob();
        } catch (error) {
            setErrorMessage(error.message);
            await loadJobs();
            await refreshSelectedJob();
        } finally {
            setLoadingLabel("");
        }
    }

    async function resumePipeline() {
        await generateSopSequentially();
    }

    async function tryLoadSop(targetJobId, showError) {
        try {
            const payload = await requestJson(`${API_BASE_URL}/jobs/${targetJobId}/sop`);
            setSopResult(payload);
            return payload;
        } catch (error) {
            if (showError) {
                setErrorMessage(error.message);
            }

            return null;
        }
    }

    function openSop() {
        if (!jobId.trim()) {
            setErrorMessage("No job selected.");
            return;
        }

        window.open(
            `${API_BASE_URL}/jobs/${jobId.trim()}/sop/markdown`,
            "_blank",
            "noopener,noreferrer",
        );
    }

    function downloadSop() {
        if (!jobId.trim()) {
            setErrorMessage("No job selected.");
            return;
        }

        window.open(
            `${API_BASE_URL}/jobs/${jobId.trim()}/sop/markdown?download=true`,
            "_blank",
            "noopener,noreferrer",
        );
    }

    return (
        <main className="app-shell">
            <div className="app-container">
                <header className="header">
                    <div>
                        <h1>Agent for Multimodal SOP generation</h1>
                        <p>
                            Upload an audio-video recording, run the full pipeline, and generate
                            a structured Standard Operating Procedure.
                        </p>
                    </div>
                </header>

                <section className="card upload-card">
                    <form className="upload-panel" onSubmit={handleUpload}>
                        <div className="section-heading">
                            <h2>Upload Video</h2>
                            <p>Create a new job run from an audio-video recording.</p>
                        </div>

                        <div className="upload-grid">
                            <div className="form-group">
                                <label htmlFor="video-file">Video file</label>
                                <input
                                    id="video-file"
                                    className="file-input"
                                    type="file"
                                    accept="video/*,audio/*"
                                    onChange={(event) => {
                                        const file = event.target.files?.[0] || null;
                                        setSelectedFile(file);
                                    }}
                                />
                            </div>

                            <div className="form-group">
                                <label htmlFor="output-filename">Output filename</label>
                                <input
                                    id="output-filename"
                                    className="text-input"
                                    type="text"
                                    value={outputFilename}
                                    onChange={(event) => setOutputFilename(event.target.value)}
                                    placeholder="generated_sop.md"
                                />
                            </div>
                        </div>

                        <div className="form-footer">
                            <div className="form-options">
                                <label className="checkbox-row">
                                    <input
                                        type="checkbox"
                                        checked={enableDiarization}
                                        onChange={(event) =>
                                            setEnableDiarization(event.target.checked)
                                        }
                                    />
                                    Enable diarization (slow; runs on full recording when enabled)
                                </label>

                                <label className="checkbox-row">
                                    <input
                                        type="checkbox"
                                        checked={extractScreenshots}
                                        onChange={(event) =>
                                            setExtractScreenshots(event.target.checked)
                                        }
                                    />
                                    Extract screenshots
                                </label>
                            </div>

                            <button className="button" type="submit" disabled={isBusy}>
                                Upload Video
                            </button>
                        </div>
                    </form>
                </section>

                <section className="card jobs-card">
                    <div className="panel-heading">
                        <div>
                            <h2>Job Runs</h2>
                            <p>Select an existing job run to view status or resume processing.</p>
                        </div>

                        <div className="panel-actions">
                            <span className="count-pill">{jobs.length}</span>
                            <button
                                className="button ghost"
                                type="button"
                                onClick={toggleAllJobSelections}
                                disabled={isBusy || jobs.length === 0}
                            >
                                {selectedJobIds.length === jobs.length && jobs.length > 0
                                    ? "Clear Selection"
                                    : "Select All"}
                            </button>
                            <button
                                className="button danger"
                                type="button"
                                onClick={() => setShowDeleteConfirm(true)}
                                disabled={isBusy || selectedJobIds.length === 0}
                            >
                                Delete Selected
                            </button>
                            <button
                                className="button ghost"
                                type="button"
                                onClick={loadJobs}
                                disabled={isBusy}
                            >
                                Refresh
                            </button>
                        </div>
                    </div>

                    <div className="jobs-list">
                        {jobs.length === 0 ? (
                            <div className="empty-state">No job runs found yet.</div>
                        ) : (
                            jobs.map((job) => {
                                const status = job.pipeline_status || "uploaded";

                                return (
                                    <div
                                        key={job.job_id}
                                        className={`job-row ${status} ${job.job_id === jobId ? "selected" : ""}`}
                                    >
                                        <label className="job-select-box" title="Select job run for deletion">
                                            <input
                                                type="checkbox"
                                                checked={selectedJobIdSet.has(job.job_id)}
                                                onChange={() => toggleJobSelection(job.job_id)}
                                                disabled={isBusy}
                                            />
                                        </label>

                                        <button
                                            type="button"
                                            className="job-row-content"
                                            onClick={() => selectJob(job)}
                                            disabled={isBusy}
                                        >
                                            <span className="job-main">
                                                <span className="job-title">{getJobDisplayName(job)}</span>
                                                <span className={`job-status-badge ${status}`}>
                                                    {getPipelineStatusLabel(status)}
                                                </span>
                                            </span>

                                            <span className="job-subtext">{getJobSubText(job)}</span>

                                            {job.pipeline_error ? (
                                                <span className="job-error">{job.pipeline_error}</span>
                                            ) : null}
                                        </button>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </section>

                <section className="card pipeline-card">
                    <div className="pipeline-header">
                        <div>
                            <h2>Pipeline</h2>
                            <p>
                                {jobId
                                    ? `Current job: ${jobId}`
                                    : "Upload a video or select an existing job run."}
                            </p>
                        </div>

                        <div className="pipeline-actions">
                            <button
                                className="button secondary"
                                type="button"
                                onClick={generateSopSequentially}
                                disabled={!canRunJobActions || isBusy}
                            >
                                Generate SOP
                            </button>

                            {failedStep || nextRunnableStep ? (
                                <button
                                    className="button ghost"
                                    type="button"
                                    onClick={resumePipeline}
                                    disabled={!canRunJobActions || isBusy}
                                >
                                    Resume
                                </button>
                            ) : null}
                        </div>
                    </div>

                    <PipelineStatusBar stepState={stepState} />

                    {loadingLabel ? (
                        <div className="notice amber">
                            Running: {loadingLabel}. CPU-heavy steps such as diarization may
                            take several minutes.
                        </div>
                    ) : null}

                    {successMessage ? (
                        <div className="notice green">{successMessage}</div>
                    ) : null}

                    {errorMessage ? (
                        <div className="notice red">{errorMessage}</div>
                    ) : null}

                    {failedStep ? (
                        <div className="failure-box">
                            <strong>Failed step:</strong> {failedStep.fullLabel}
                            <br />
                            <strong>Error:</strong>{" "}
                            {stepState[failedStep.key]?.error || "Unknown error"}
                        </div>
                    ) : null}
                </section>

                <section className="card sop-card">
                    <div className="sop-header">
                        <div>
                            <h2>Generated SOP</h2>
                            <p>
                                The SOP appears here after generation. Use Open to view it in a
                                browser tab or Download to save the markdown file.
                            </p>
                        </div>

                        <div className="sop-actions">
                            <button
                                className="icon-button"
                                type="button"
                                onClick={openSop}
                                disabled={!hasGeneratedSop || !jobId}
                                title="Open SOP"
                            >
                                ↗ Open
                            </button>

                            <button
                                className="icon-button"
                                type="button"
                                onClick={downloadSop}
                                disabled={!hasGeneratedSop || !jobId}
                                title="Download SOP"
                            >
                                ⬇ Download
                            </button>
                        </div>
                    </div>

                    <div className="sop-box">
                        {sopResult
                            ? extractSopText(sopResult)
                            : "No SOP generated for the selected job yet."}
                    </div>
                </section>
            </div>

            {showDeleteConfirm ? (
                <div className="modal-backdrop" role="presentation">
                    <div className="modal-card" role="dialog" aria-modal="true" aria-labelledby="delete-jobs-title">
                        <h2 id="delete-jobs-title">Delete selected job runs?</h2>
                        <p>
                            This will permanently delete the selected job run records and all related files,
                            including uploaded videos, extracted audio, frames, OCR, transcripts, timeline,
                            activities, refined activities, diarization output, and generated SOP files.
                        </p>

                        <div className="delete-list-box">
                            {selectedJobsForDelete.map((job) => (
                                <div className="delete-list-item" key={job.job_id}>
                                    <strong>{getJobDisplayName(job)}</strong>
                                    <span>{job.job_id}</span>
                                </div>
                            ))}
                        </div>

                        <div className="modal-actions">
                            <button
                                className="button ghost"
                                type="button"
                                onClick={() => setShowDeleteConfirm(false)}
                                disabled={isBusy}
                            >
                                Cancel
                            </button>
                            <button
                                className="button danger"
                                type="button"
                                onClick={deleteSelectedJobs}
                                disabled={isBusy || selectedJobIds.length === 0}
                            >
                                Delete Permanently
                            </button>
                        </div>
                    </div>
                </div>
            ) : null}
        </main>
    );
}