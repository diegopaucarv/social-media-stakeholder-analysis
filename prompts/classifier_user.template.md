<post_context>
{{POST_CONTEXT}}
</post_context>

<comments_batch>
{{COMMENTS_BATCH}}
</comments_batch>

<instructions>
Classify ALL comments in <comments_batch> following the codebook, decision rules and output schema of the system prompt. Return only the JSON array, one object per comment, in the same order as the input.
</instructions>
