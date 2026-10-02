import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "match_state", "label": "Match state (JSON)", "type": "object", "default": {"runs_needed": 50, "balls_remaining": 30, "wickets_remaining": 6}}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'sports-analytics-hub',Controls});
