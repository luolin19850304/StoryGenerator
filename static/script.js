const STORY = document.getElementById('story');
const BTN_SEND = document.querySelector('.btn[type=submit]');

function setStoryText(txt) {
  STORY.innerText = txt;
}

function notifyLoad() {
  BTN_SEND.innerText = 'Wait';
  BTN_SEND.classList.remove('btn-primary');
  BTN_SEND.classList.add('btn-light');
  BTN_SEND.setAttribute('disabled', 'true');
}

function unNotifyLoad() {
  BTN_SEND.innerText = 'Create';
  BTN_SEND.removeAttribute('disabled');
  BTN_SEND.classList.add('btn-primary');
  BTN_SEND.classList.remove('btn-light');
}

document.querySelector('.btn[type=submit]').onclick = function() {
  notifyLoad();
  return fetch(`/`, {
    method: 'post',
    body: JSON.stringify({
      seed: document.querySelector('textarea').value,
      algo: document.getElementById('picker-algo').value,
      n: parseInt(document.getElementById('picker-n').value),
      len: parseInt(document.getElementById('picker-len').value),
    }),
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/plain',
    },
  }).then(res => res.status >= 400
      ? Promise.reject(setStoryText(res.statusText))
      : res.text(),
  ).then(setStoryText).then(unNotifyLoad);
};
