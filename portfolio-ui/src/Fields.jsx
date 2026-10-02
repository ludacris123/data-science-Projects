import {React} from './App.jsx';
export function Fields({fields,data,update}) {
  return <section><h3>Project controls</h3>{fields.map(field => {
    const value=data[field.key] ?? field.default ?? '';
    return <label key={field.key}>{field.label}{field.options ? <select value={value} onChange={event=>update(field.key,event.target.value)}>{field.options.map(option=><option key={option} value={option}>{option}</option>)}</select> : field.type==='object' ? <textarea aria-label={field.label} defaultValue={JSON.stringify(value,null,2)} onBlur={event=>{try{update(field.key,JSON.parse(event.target.value));event.target.setCustomValidity('')}catch{event.target.setCustomValidity('Enter valid JSON');event.target.reportValidity()}}}/> : <input type={field.type || 'text'} min={field.min} max={field.max} value={value} onChange={event=>{const raw=event.target.value;update(field.key,field.type==='number'&&raw!==''?Number(raw):raw)}}/>}</label>;
  })}</section>;
}
