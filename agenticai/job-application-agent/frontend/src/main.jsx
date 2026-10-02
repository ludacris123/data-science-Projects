import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "query", "label": "Job preferences"}, {"key": "resume", "label": "Resume text"}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'job-application-agent',Controls});
